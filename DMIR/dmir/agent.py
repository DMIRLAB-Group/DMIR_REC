import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from memory import RealCounterFactualReplayBuffer as Buffer
from model import TEA_state


class StepwiseLR:

    def __init__(self, optimizer, init_lr,
                 gamma, decay_rate):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        lr = self.get_lr()
        self.iter_num += 1
        for param_group in self.optimizer.param_groups:
            if "lr_mult" not in param_group:
                param_group["lr_mult"] = 1
            param_group['lr'] = lr * param_group["lr_mult"]


class DMIR:
    def __init__(self, cfg):
        state_dim = cfg.edim
        action_dim = cfg.item_num
        self.user_num = cfg.user_num
        self.update_size = cfg.update_size
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.frame_idx = 0
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = TEA_state(self.user_num, action_dim, state_dim, cfg.droprate, cfg.use_pos_emb, cfg.seq_len,
                                    cfg.device, cfg.use_att, cfg.nbr_wei, cfg).to(
            self.device)
        self.target_net = TEA_state(self.user_num, action_dim, state_dim, cfg.droprate, cfg.use_pos_emb, cfg.seq_len,
                                    cfg.device, cfg.use_att, cfg.nbr_wei, cfg).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer = optim.Adam([{'params': self.policy_net.parameters()}],
                                    lr=cfg.lr)
        self.lr_scheduler = StepwiseLR(self.optimizer, init_lr=cfg.lr, gamma=cfg.lr_gamma, decay_rate=cfg.lr_decay_rate)
        self.buffer = Buffer(cfg.buffer_size)

    def choose_action(self, user, memory, memory_rate, nbr, nbr_iid, nbr_iid_rate, available_items, ava_item_mask,
                      is_done, valid=False):

        if valid == True or random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = self.policy_net((user, memory, memory_rate, nbr, nbr_iid, nbr_iid_rate))[:, -1, :].unsqueeze(
                    1)
                item_embeddings = self.policy_net.item_embs(available_items)
                state = state.expand(-1, item_embeddings.shape[1], -1)

                item_embeddings = item_embeddings.repeat(1, 1, 2)
                q_values = torch.sum(state * item_embeddings, dim=-1)
                e_q_values = torch.exp(q_values)

                e_q_values[available_items == 0] = -9999999999

                action_id = e_q_values.max(1)[1].unsqueeze(-1)
                action = available_items.gather(-1, action_id)
        else:
            action_id = []
            max_item = available_items.shape[1]
            for i in range(user.shape[0]):

                if is_done[i] == True:
                    action_id.append(0)
                else:
                    tmp = np.random.randint(low=0, high=max_item)
                    while available_items[i, tmp] == 0:
                        tmp = np.random.randint(low=0, high=max_item)
                    action_id.append(tmp)
            action_id = torch.tensor(action_id).unsqueeze(-1).to(self.device)
            action = available_items.gather(-1, action_id)
        return action_id.to('cpu'), action.to('cpu')

    def update(self, available_items, all_nbr, all_nbr_iid, all_nbr_iid_rate, args):
        self.policy_net.train()
        self.target_net.train()
        user_batch, memory_batch, memory_rate_batch, action_batch, reward_batch, next_memory_batch, \
        next_memory_rate_batch, done_batch, mask_batch = self.buffer.sample(self.update_size)
        if args.replay == 0:
            self.buffer.rest()

        user_batch = torch.tensor(user_batch)
        nbr_batch = all_nbr[user_batch].to(self.device)
        nbr_iid_batch = all_nbr_iid[user_batch].to(self.device)
        nbr_iid_rate_batch = all_nbr_iid_rate[user_batch].to(self.device)
        user_batch = user_batch.to(self.device)
        memory_batch = torch.cat(memory_batch, 0).reshape(self.update_size, -1).to(self.device)
        memory_rate_batch = torch.cat(memory_rate_batch, 0).reshape(self.update_size, -1).to(self.device)
        action_batch = torch.tensor(action_batch).to(self.device)
        reward_batch = torch.tensor(reward_batch, device=self.device)
        next_memory_batch = torch.cat(next_memory_batch, 0).reshape(self.update_size, -1).to(self.device)
        next_memory_rate_batch = torch.cat(next_memory_rate_batch, 0).reshape(self.update_size, -1).to(self.device)
        done_batch = torch.tensor(done_batch, device=self.device)
        mask_batch = torch.cat(mask_batch, 0).reshape(self.update_size, -1).to(self.device)

        state_batch = self.policy_net(
            (user_batch, memory_batch, memory_rate_batch, nbr_batch, nbr_iid_batch, nbr_iid_rate_batch))[:, -1, :]
        item_embeddings_batch = self.policy_net.item_embs(action_batch)

        item_embeddings_batch = item_embeddings_batch.repeat(1, 2)
        q_values = torch.exp(torch.sum(state_batch * item_embeddings_batch, -1))

        ava_items_batch = available_items.to(self.device)[user_batch]
        ava_items_embedding_batch = self.policy_net.item_embs(ava_items_batch)
        next_state_batch = self.policy_net((user_batch, next_memory_batch, next_memory_rate_batch, nbr_batch,
                                            nbr_iid_batch, nbr_iid_rate_batch))[:, -1, :].unsqueeze(1). \
            expand(-1, ava_items_embedding_batch.shape[1], -1)

        ava_items_embedding_batch = ava_items_embedding_batch.repeat(1, 1, 2)
        next_q_values = torch.exp(torch.sum(next_state_batch * ava_items_embedding_batch, -1))

        next_q_values[ava_items_batch == 0] = -9999999999

        tar_ava_items_embedding_batch = self.target_net.item_embs(ava_items_batch)

        tar_next_state_batch = self.target_net((user_batch, next_memory_batch, next_memory_rate_batch, nbr_batch,
                                                nbr_iid_batch, nbr_iid_rate_batch))[:, -1, :].unsqueeze(1). \
            expand(-1, ava_items_embedding_batch.shape[1], -1)

        tar_ava_items_embedding_batch = tar_ava_items_embedding_batch.repeat(1, 1, 2)
        tar_next_q_values = torch.exp(
            torch.sum(tar_next_state_batch * tar_ava_items_embedding_batch, -1))

        tar_next_q_values = tar_next_q_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(-1)

        expected_q_values = reward_batch + self.gamma * tar_next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def save(self, args, path):
        torch.save(self.target_net.state_dict(), path + f'GT{args.edim}dmir_target_net_checkpoint.pth')
        torch.save(self.policy_net.state_dict(), path + f'GT{args.edim}dmir_policy_net_checkpoint.pth')

    def load(self, args, path):
        self.target_net.load_state_dict(torch.load(path + f'GT{args.edim}dmir_target_net_checkpoint.pth'))
        self.policy_net.load_state_dict(torch.load(path + f'GT{args.edim}dmir_policy_net_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)

