
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

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

class TEA_state(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 edim,
                 droprate,
                 use_pos_emb,
                 seq_maxlen,
                 device,
                 use_att,
                 nbr_wei,
                 args):
        super(TEA_state, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.edim = edim
        self.droprate = droprate
        self.dev = torch.device(device)
        self.use_pos_emb = use_pos_emb
        self.seq_maxlen = seq_maxlen
        self.use_att = use_att
        num_heads = 1

        self.item_attn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.item_attn_layer = nn.MultiheadAttention(edim, num_heads, droprate)
        self.item_ffn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.item_ffn = FFN(edim, droprate)
        self.item_last_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.seq_lin = nn.Linear(edim + edim, edim)

        self.rnn = nn.GRU(input_size=edim, hidden_size=edim, num_layers=1, batch_first=True)

        self.seq_rate_fuse = nn.Linear(edim + edim, edim)
        self.nbr_rate_fuse = nn.Linear(edim + edim, edim)
        self.nbr_item_fsue_lin = nn.Linear(edim + edim, edim)
        self.nbr_ffn_layernom = nn.LayerNorm(edim, eps=1e-8)
        self.nbr_ffn = FFN(edim, droprate)
        self.nbr_last_layernorm = nn.LayerNorm(edim, eps=1e-8)

        self.user_embs = nn.Embedding(user_num, edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        self.tar_item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        self.posn_embs = nn.Embedding(seq_maxlen, edim, padding_idx=0)
        self.rate_embs = nn.Embedding(2, edim)
        nn.init.uniform_(self.user_embs.weight[1:], a=-0.5/user_num, b=0.5/user_num)
        nn.init.uniform_(self.item_embs.weight[1:], a=-0.5/item_num, b=0.5/item_num)
        nn.init.uniform_(self.tar_item_embs.weight[1:], a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.posn_embs.weight[1:], a=-0.5/seq_maxlen, b=0.5/seq_maxlen)
        nn.init.uniform_(self.rate_embs.weight, a=-0.5 / 2, b=0.5 / 2)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(droprate)
        self.nbr_wei = nbr_wei
        self.neg_wei = args.neg_wei
        self.args=args

    def seq2feat(self, seq_iid):
        timeline_mask = torch.BoolTensor(seq_iid.to('cpu') == 0).to(self.dev)
        seqs = self.item_embs(seq_iid.to(self.dev)) * (self.item_embs.embedding_dim ** 0.5)
        if self.use_pos_emb:
            positions = np.tile(np.array(range(seq_iid.shape[1]), dtype=np.int64), [seq_iid.shape[0], 1])
            seqs += self.posn_embs(torch.LongTensor(positions).to(self.dev))

        seqs = self.dropout(seqs)

        seqs *= ~timeline_mask.unsqueeze(-1)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        seqs = torch.transpose(seqs, 0, 1)
        query = self.item_attn_layernorm(seqs)
        mha_outputs, _ = self.item_attn_layer(query, seqs, seqs, attn_mask=attention_mask)

        seqs = query + mha_outputs
        seqs = torch.transpose(seqs, 0, 1)
        seqs = self.item_ffn_layernorm(seqs)
        seqs = self.item_ffn(seqs)
        seqs *= ~timeline_mask.unsqueeze(-1) 
        seqs = self.item_last_layernorm(seqs)
        if self.use_att == 1:
            last_seq = seqs[:,-1,:].clone().unsqueeze(1)
            att = (last_seq*seqs).sum(-1)
            att *= ~timeline_mask
            att = torch.softmax(att,dim=-1)
            seqs = (seqs*att.unsqueeze(-1)).sum(-2, keepdim=True)
            seqs = seqs.expand(-1, seq_iid.shape[-1], -1)
        return seqs

    def nbr2feat(self, nbr, nbr_iid):

        batch_size, nbr_maxlen, seq_maxlen = nbr_iid.shape
        nbr_mask = torch.BoolTensor(nbr.to('cpu') == 0).to(self.dev)
        nbr_seq_mask = torch.BoolTensor(nbr_iid.to('cpu') == 0).to(self.dev)
        nbr_len = (nbr_maxlen - nbr_mask.sum(1)) 
        nbr_len[torch.where(nbr_len == 0)] = 1.0


        nbr = nbr.to(self.dev)
        nbr_iid = nbr_iid.to(self.dev)
        nbr_emb = self.dropout(self.user_embs(nbr))
        nbr_item_emb = self.dropout(self.item_embs(nbr_iid))

        nbr_emb *= ~nbr_mask.unsqueeze(-1)
        nbr_len = nbr_len.view(batch_size, 1, 1)
        nbr_feat = nbr_emb.sum(dim=1, keepdim=True) / nbr_len

        nbr_seq_mask = nbr_seq_mask.unsqueeze(-1)
        nbr_seq_mask = nbr_seq_mask.permute(0, 2, 1, 3)
        nbr_item_emb = nbr_item_emb.permute(0, 2, 1, 3)
        nbr_item_emb *= ~nbr_seq_mask
        nbr_seq_len = (20 - nbr_seq_mask.sum(dim=2))

        nbr_seq_len[torch.where(nbr_seq_len == 0)] = 1.0
        nbr_seq_feat = nbr_item_emb.sum(dim=2) / nbr_seq_len
        nbr_seq_feat, _ = self.rnn(nbr_seq_feat)

        nbr_feat = nbr_feat.expand_as(nbr_seq_feat)
        nbr_feat = self.nbr_item_fsue_lin(torch.cat([nbr_feat, nbr_seq_feat], dim=-1))
        nbr_feat = self.nbr_ffn_layernom(nbr_feat)
        nbr_feat = self.nbr_ffn(nbr_feat)
        nbr_feat = self.nbr_last_layernorm(nbr_feat)

        if self.use_att == 1:
            last_seq = nbr_feat[:,-1,:].clone().unsqueeze(1)
            att = (last_seq*nbr_feat).sum(-1)
            att = torch.softmax(att, dim=-1)
            nbr_feat = (nbr_feat*att.unsqueeze(-1)).sum(-2,keepdim=True)
            nbr_feat = nbr_feat.expand(-1, nbr_iid.shape[-2], -1)
        return nbr_feat

    def dual_pred(self, seq_hu, nbr_hu, hi):
        seq_logits = (seq_hu * hi).sum(dim=-1)
        nbr_logits = (nbr_hu * hi).sum(dim=-1)
        return seq_logits + nbr_logits

    def forward(self, batch):
        uid, seq, seq_rate, nbr, nbr_iid, nbr_iid_rate = batch
        pos_seq = seq.clone()
        neg_seq = seq.clone()
        pos_nbr_iid = nbr_iid.clone()
        neg_nbr_iid = nbr_iid.clone()

        pos_seq[seq_rate==0] = 0
        neg_seq[seq_rate>0] = 0
        pos_nbr_iid[nbr_iid_rate==0] = 0
        neg_nbr_iid[nbr_iid_rate>0] = 0

        uid = uid.unsqueeze(-1).expand_as(seq)
        user_emb = self.dropout(self.user_embs(uid.to(self.dev)))

        pos_seq_feat = self.seq2feat(pos_seq)
        pos_seq_feat = self.seq_lin(torch.cat([pos_seq_feat, user_emb], dim=-1))
        neg_seq_feat = self.seq2feat(neg_seq)
        neg_seq_feat = self.seq_lin(torch.cat([neg_seq_feat, user_emb], dim=-1))

        pos_nbr_feat = self.nbr_wei*self.nbr2feat(nbr, pos_nbr_iid)
        neg_nbr_feat = self.nbr_wei*self.nbr2feat(nbr, neg_nbr_iid)
        if self.args.use_nbr==0:
            pos_feat = torch.cat((pos_seq_feat, pos_seq_feat), dim=-1)
            neg_feat = self.neg_wei*torch.cat((neg_seq_feat, neg_seq_feat), dim=-1)
        else:
            pos_feat = torch.cat((pos_seq_feat, pos_nbr_feat), dim=-1)
            neg_feat = self.neg_wei * torch.cat((neg_seq_feat, neg_nbr_feat), dim=-1)

        return pos_feat-neg_feat

    def get_parameters(self):

        param_list = [
            {'params': self.item_attn_layernorm.parameters()},
            {'params': self.item_attn_layer.parameters()},
            {'params': self.item_ffn_layernorm.parameters()},
            {'params': self.item_ffn.parameters()},
            {'params': self.item_last_layernorm.parameters()},

            {'params': self.seq_lin.parameters()},
            {'params': self.rnn.parameters()},
            {'params': self.seq_rate_fuse.parameters()},
            {'params': self.nbr_rate_fuse.parameters()},
            {'params': self.nbr_item_fsue_lin.parameters()},
            {'params': self.nbr_ffn_layernom.parameters()},
            {'params': self.nbr_ffn.parameters()},
            {'params': self.nbr_last_layernorm.parameters()},

            {'params': self.user_embs.parameters(), 'weight_decay': 0},
            {'params': self.item_embs.parameters(), 'weight_decay': 0},
            {'params': self.posn_embs.parameters(), 'weight_decay': 0},
            {'params': self.rate_embs.parameters(), 'weight_decay': 0},
        ]


        return param_list