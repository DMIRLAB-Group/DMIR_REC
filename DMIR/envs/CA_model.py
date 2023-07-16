import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(FFN, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class CA(nn.Module):
    def __init__(self, user_num, item_num, Z_top, edim, droprate, dataset, seq_len, device, args):
        super(CA, self).__init__()
        self.user_emb = nn.Embedding(user_num, edim, padding_idx=0)
        self.item_emb = nn.Embedding(item_num, edim, padding_idx=0)
        self.rate_emb = nn.Embedding(2, edim // args.rate_edim_pa)
        nn.init.uniform_(self.user_emb.weight[1:], a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_emb.weight[1:], a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.rate_emb.weight, a=-0.5 / 2, b=0.5 / 2)

        self.beta_fuse_linear = nn.Linear(edim + edim // args.rate_edim_pa + edim // args.Z_edim_pa, edim)
        self.beta_nbr_item_fuse_linear = nn.Linear(edim + edim, edim)
        self.beta_rnn = nn.GRU(input_size=edim, hidden_size=edim, num_layers=1, batch_first=True)
        self.beta_item_attn_layernorm = nn.LayerNorm(edim)
        self.beta_item_attn_layer = nn.MultiheadAttention(edim, 1, droprate)
        self.beta_mean_linear = nn.Linear(edim * 2, edim // args.beta_edim_pa)
        self.beta_std_linear = nn.Linear(edim * 2, edim // args.beta_edim_pa)
        self.beta_ffn = FFN(edim, args.droprate)
        self.beta_ffn_layernorm = nn.LayerNorm(edim)

        self.U_item_Z_linear = nn.Linear(edim + edim // args.Z_edim_pa, edim)
        self.U_self_fuse_linear = nn.Linear(edim + edim, edim)
        self.U_rnn = nn.GRU(input_size=edim, hidden_size=edim, num_layers=1, batch_first=True)
        self.U_self_item_attn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.U_self_item_attn_layer = nn.MultiheadAttention(edim, 1, droprate)
        self.U_nbr_rate_linear = nn.Linear(edim + edim // args.rate_edim_pa, edim)
        self.U_nbr_item_fuse_linear = nn.Linear(edim + edim, edim)
        self.U_linear = nn.Linear(edim * 2, edim)
        self.U_self_ffn = FFN(edim, args.droprate)
        self.U_self_ffn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.U_nbr_ffn = FFN(edim, args.droprate)
        self.U_nbr_ffn_layernorm = nn.LayerNorm(edim, eps=1e-8)

        self.act = nn.Sigmoid()

        self.de_rate_linear = nn.Linear(edim * 2 + edim // args.beta_edim_pa + edim // args.Z_edim_pa, 1)

        self.Z_top = Z_top
        self.dev = device
        self.dropout = nn.Dropout(droprate)
        self.edim = edim
        self.Z_edim_pa = args.Z_edim_pa
        self.dataset = dataset
        self.fin_U = torch.zeros((user_num, seq_len, edim))
        self.fin_beta = torch.zeros((user_num, edim // 8))
        self.loss_fun = torch.nn.MSELoss()

    def get_beta(self, user, seq, Z, rate, U):
        batch_size = seq.shape[0]
        Z = Z.unsqueeze(-1)
        seq_emb = self.dropout(self.item_emb(seq))
        rate_emb = self.dropout(self.rate_emb(rate))
        seq_mask = torch.BoolTensor(seq.to('cpu') != 0).to(self.dev)
        cat = torch.cat((seq_emb, rate_emb, Z.expand(-1, -1, self.edim // self.Z_edim_pa)), -1)
        cat = self.dropout(self.beta_fuse_linear(cat))
        cat *= seq_mask.unsqueeze(-1)
        tl = seq.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        cat = cat.transpose(0, 1)

        qurey = self.beta_item_attn_layernorm(cat)
        att_out, _ = self.beta_item_attn_layer(qurey, cat, cat, attn_mask=attention_mask)
        cat = cat + att_out
        cat = cat.transpose(0, 1)
        cat = self.beta_ffn(cat)
        cat = self.beta_ffn_layernorm(cat)

        beta_mean = self.beta_mean_linear(torch.cat((cat, U), -1)).mean(1)
        beta_std = self.beta_std_linear(torch.cat((cat, U), -1)).mean(1)

        self.fin_beta[user] = beta_mean.to('cpu')

        return beta_mean, beta_std

    def get_U(self, user, seq, Z, nbr, nbr_iid, nbr_rate):
        Z = Z.unsqueeze(-1)
        seq_emb = self.dropout(self.item_emb(seq))
        seq_emb = self.U_item_Z_linear(torch.cat((seq_emb, Z.expand(-1, -1, self.edim // self.Z_edim_pa)), dim=-1))
        seq_mask = torch.BoolTensor(seq.to('cpu') != 0).to(self.dev).unsqueeze(-1)
        seq_emb *= seq_mask
        tl = seq.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        seq_emb = seq_emb.transpose(0, 1)
        qurey = self.U_self_item_attn_layernorm(seq_emb)
        att_out, _ = self.U_self_item_attn_layer(qurey, seq_emb, seq_emb, attn_mask=attention_mask)
        seq_emb = seq_emb + att_out
        seq_emb = seq_emb.transpose(0, 1)
        seq_emb = self.U_self_ffn(seq_emb)
        seq_emb = self.U_self_ffn_layernorm(seq_emb)

        user_emb = self.dropout(self.user_emb(user)).unsqueeze(1).expand_as(seq_emb)
        cat = self.U_self_fuse_linear(torch.cat((user_emb, seq_emb), dim=-1))

        nbr_mask = torch.BoolTensor(nbr.to('cpu') != 0).unsqueeze(-1).to(self.dev)
        nbr_len = nbr_mask.sum(1, keepdim=True)
        nbr_len[nbr_len == 0] = 1
        nbr_emb = self.dropout(self.user_emb(nbr))
        nbr_emb *= nbr_mask
        nbr_feat = nbr_emb.sum(1, keepdim=True) / nbr_len

        nbr_seq_mask = torch.BoolTensor(nbr_iid.to('cpu') != 0).unsqueeze(-1).to(self.dev)
        nbr_seq_len = nbr_seq_mask.sum(1, keepdim=True)
        nbr_seq_len[nbr_seq_len == 0] = 1
        nbr_iid_emb = self.item_emb(nbr_iid)
        nbr_rate_emb = self.rate_emb(nbr_rate)
        nbr_seq_cat = torch.cat((nbr_iid_emb, nbr_rate_emb), -1)
        nbr_seq_cat_emb = self.dropout(self.U_nbr_rate_linear(nbr_seq_cat))
        nbr_seq_cat_emb *= nbr_seq_mask
        nbr_seq_feat = nbr_seq_cat_emb.sum(1, keepdim=True) / nbr_seq_len
        nbr_seq_feat = nbr_seq_feat.squeeze(1)
        nbr_seq_feat, _ = self.U_rnn(nbr_seq_feat)

        nbr_feat = nbr_feat.expand_as(nbr_seq_feat)
        G = self.U_nbr_item_fuse_linear(torch.cat([nbr_feat, nbr_seq_feat], -1))
        G = self.U_nbr_ffn(G)
        G = self.U_nbr_ffn_layernorm(G)

        U = self.U_linear(torch.cat((cat, G), -1))

        self.fin_U[user] = U.to('cpu')

        return U

    def reparamized(self, mean, std):
        return mean + torch.exp(0.5 * std) * torch.rand(mean.shape).to(self.dev)

    def de_rate(self, beta, Z, seq, U):
        seq_emb = self.dropout(self.item_emb(seq))
        beta = beta.unsqueeze(-2).expand(-1, seq.shape[1], -1)
        Z = Z.unsqueeze(-1)
        pre_rate = self.de_rate_linear(
            torch.cat((seq_emb, U, beta, Z.expand(-1, -1, self.edim // self.Z_edim_pa)), -1))

        return pre_rate

    def get_rate_loss(self, pre_rate, label):
        pre_rate = pre_rate.reshape(-1)
        tmp = torch.isnan(pre_rate)
        lo = torch.where(tmp > 0)
        label = label.reshape(-1).float()
        return F.binary_cross_entropy_with_logits(pre_rate, label)

    def forward(self, batch, ind=None):
        user, seq, rate, nbr, nbr_iid, nbr_iid_rate, seq_Z = batch
        user = user.to(self.dev)
        seq = seq.to(self.dev)
        rate = rate.to(self.dev)
        nbr = nbr.to(self.dev)
        nbr_iid = nbr_iid.to(self.dev)
        nbr_iid_rate = nbr_iid_rate.to(self.dev)
        seq_Z = seq_Z.to(self.dev)

        U = self.get_U(user, seq, seq_Z, nbr, nbr_iid, nbr_iid_rate)
        beta_mean, beta_std = self.get_beta(user, seq, seq_Z, rate, U)
        beta = self.reparamized(beta_mean, beta_std)

        de_rate = self.de_rate(beta, seq_Z, seq, U)
        if ind != None:
            de_rate_loss = self.get_rate_loss(de_rate[ind], rate[ind])
        else:
            de_rate_loss = self.get_rate_loss(de_rate, rate)
        beta_KL_divergence = 0.5 * torch.mean(-beta_std + torch.square(beta_mean) + torch.exp(beta_std) - 1)
        loss = de_rate_loss + beta_KL_divergence
        return loss

    def inference(self, beta, user, seq, nbr, nbr_iid, nbr_iid_rate, seq_Z):
        with torch.no_grad():
            beta = beta.to(self.dev)
            user = user.to(self.dev)
            seq = seq.to(self.dev)
            nbr = nbr.to(self.dev)
            nbr_iid = nbr_iid.to(self.dev)
            nbr_iid_rate = nbr_iid_rate.to(self.dev)
            seq_Z = seq_Z.to(self.dev)

            U = self.get_U(user, seq, seq_Z, nbr, nbr_iid, nbr_iid_rate)

            pre_rate = self.de_rate(beta, seq_Z[:, -1].unsqueeze(-1), seq[:, -1].unsqueeze(-1),
                                    U[:, -1, :].unsqueeze(-2)).squeeze(-2)

            return pre_rate

    def train_inference(self, beta, Z, seq, U):
        self.eval()
        with torch.no_grad():
            batch_size = U.shape[0]
            beta = beta.to(self.dev)
            Z = Z.to(self.dev)
            seq = seq.to(self.dev)
            U = U.to(self.dev)
            seq_emb = self.item_emb(seq)
            U = U.expand(batch_size, seq_emb.shape[1], -1)
            Z = Z.unsqueeze(-1)
            beta = beta.unsqueeze(1).expand(batch_size, seq_emb.shape[1], -1)
            pre_rate = self.de_rate_linear(
                torch.cat((seq_emb, U, beta, Z.expand(-1, -1, self.edim // self.Z_edim_pa)), -1))

            pre_rate = self.act(pre_rate)
            return pre_rate

    def valid_inference(self, beta, Z, seq, U):
        self.eval()
        with torch.no_grad():
            batch_size = U.shape[0]
            beta = beta.to(self.dev)
            Z = Z.to(self.dev)
            seq = seq.to(self.dev)
            U = U.to(self.dev)
            seq_emb = self.item_emb(seq)
            U = U.unsqueeze(1).expand(batch_size, seq_emb.shape[1], -1)
            Z = Z.unsqueeze(-1)
            beta = beta.unsqueeze(1).expand(batch_size, seq_emb.shape[1], -1)
            pre_rate = self.de_rate_linear(
                torch.cat((seq_emb, U, beta, Z.expand(-1, -1, self.edim // self.Z_edim_pa)), -1))

            pre_rate = self.act(pre_rate)
            return pre_rate

    def get_parameters(self):
        param_list = [
            {'params': self.beta_fuse_linear.parameters()},
            {'params': self.beta_nbr_item_fuse_linear.parameters()},
            {'params': self.beta_rnn.parameters()},
            {'params': self.beta_item_attn_layer.parameters()},
            {'params': self.beta_mean_linear.parameters()},
            {'params': self.beta_std_linear.parameters()},
            {'params': self.beta_ffn.parameters()},

            {'params': self.U_item_Z_linear.parameters()},
            {'params': self.U_self_fuse_linear.parameters()},
            {'params': self.U_rnn.parameters()},
            {'params': self.U_self_item_attn_layer.parameters()},
            {'params': self.U_nbr_rate_linear.parameters()},
            {'params': self.U_nbr_item_fuse_linear.parameters()},
            {'params': self.U_linear.parameters()},

            {'params': self.U_self_ffn.parameters()},
            {'params': self.U_nbr_ffn.parameters()},
            {'params': self.de_rate_linear.parameters()},

            {'params': self.user_emb.parameters(), 'weight_decay': 0},
            {'params': self.item_emb.parameters(), 'weight_decay': 0},
            {'params': self.rate_emb.parameters(), 'weight_decay': 0}
        ]

        return param_list





