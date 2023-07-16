import torch
import numpy as np
import os, sys
sys.path.append('../')
from tqdm import tqdm
import time
from envs.CA_model import CA

from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED

class Env():
    def __init__(self, user_item_mat, user_train,
                 all_nbr, all_nbr_iid, all_nbr_iid_rate, self_nbr_item_sets,
                 Z_d, Z_bin, Z_time_interval_size, Z_min_time, Z_max_time,
                 ts, args, updated_CA_path=''):
        self.reset(user_item_mat, user_train,
                 all_nbr, all_nbr_iid, all_nbr_iid_rate, self_nbr_item_sets,
                 Z_d, Z_bin, Z_time_interval_size, Z_min_time, Z_max_time,
                 ts, args, updated_CA_path)

    def reset(self, user_item_mat, user_train,
                 all_nbr, all_nbr_iid, all_nbr_iid_rate, self_nbr_item_sets,
                 Z_d, Z_bin, Z_time_interval_size, Z_min_time, Z_max_time,
                 ts, args, updated_CA_path=''):
        self.reward_decay = args.train_reward_decay
        self.pos_line = args.pos_line
        self.max_item = args.max_item
        self.user_item = user_train
        self.user_item_mat = user_item_mat
        self.item_count = args.item_num
        self.user_num = args.user_num
        self.item_num = args.item_num
        self.memory_size = args.memory_size
        self.memory = torch.zeros([args.user_num, self.memory_size]).long()
        self.memory_rate = torch.zeros([args.user_num, self.memory_size]).long()
        self.memory_Z = torch.zeros([args.user_num, self.memory_size]).long()
        self.memory_len = torch.zeros(args.user_num).long()
        self.available_items = [np.zeros(self.max_item)]
        self.available_pos_num = [0]
        nbr_len = all_nbr.shape[-1]
        nbr_iid_len = all_nbr_iid.shape[-1]
        self.all_nbr = torch.cat((torch.zeros(1, nbr_len), torch.from_numpy(all_nbr)), dim=0).long()
        self.all_nbr_iid = torch.cat((torch.zeros(1, nbr_len, nbr_iid_len), torch.from_numpy(all_nbr_iid)),
                                     dim=0).long()

        self.all_nbr_iid_rate = torch.cat((torch.zeros(1, nbr_len, nbr_iid_len), torch.from_numpy(all_nbr_iid_rate)),
                                          dim=0).long()
        self.all_nbr_iid_rate[self.all_nbr_iid_rate <= self.pos_line] = 0
        self.all_nbr_iid_rate[self.all_nbr_iid_rate > self.pos_line] = 1
        self.self_nbr_item_sets = self_nbr_item_sets


        self.counterfactor = CA(args.user_num, args.item_num, args.Z_top, args.CA_edim, args.CA_drop, 'dataset',
                                args.seq_len, args.device, args)
        print("path",os.getcwd())
        files = os.listdir(f'../envs/saved_models/{args.dataset}/')
        file = ''
        for ff in files:
            if ff[0]=='C':
                file = ff
        print("FILE",file)
        if updated_CA_path == '':
            if args.dataset == "Yelp":

                self.counterfactor.load_state_dict(torch.load(f'../envs/saved_models/{args.dataset}/'+file))
            elif args.dataset == "Epin":
                self.counterfactor.load_state_dict(torch.load(f'../envs/saved_models/{args.dataset}/'+file))
            elif args.dataset == "Ciao":
                self.counterfactor.load_state_dict(torch.load(f'../envs/saved_models/{args.dataset}/'+file))
        else:
            self.counterfactor.load_state_dict(torch.load(updated_CA_path))
        self.counterfactor.to(args.device)
        self.Z_d = torch.cat((torch.zeros(1, args.seq_len), torch.tensor(Z_d)), dim=0)
        self.Z_bin = Z_bin
        self.Z_bin_sum = Z_bin.sum(-1)
        self.Z_min_time = Z_min_time
        self.Z_max_time = Z_max_time
        self.Z_time_interval_size = Z_time_interval_size
        self.ts = torch.cat((torch.zeros(1, nbr_iid_len), torch.from_numpy(ts)), dim=0)
        self.seq_id = torch.zeros(args.user_num).long()
        self.true_item_num = 0
        self.counter_item_num = 0
        self.pos_item_num = 0
        self.neg_item_num = 0
        self.check_line = args.check_line
        t0 = time.time()
        print('start random')
        for user in tqdm(range(1, self.user_num)):
            real_item = self.user_item[user][:, 0]
            real_num = len(real_item)
            pos_item = self.user_item[user][np.where(self.user_item[user][:, 2] > self.pos_line), 0][0]
            pos_num = len(pos_item)
            neg_item = self.user_item[user][np.where(self.user_item[user][:, 2] <= self.pos_line), 0][0]
            neg_num = len(neg_item)
            if pos_num < self.max_item:
                self.available_pos_num.append(pos_num)
            else:
                self.available_pos_num.append(self.max_item)
            nbr_item_set = self.self_nbr_item_sets[user].copy()
            for tmp in real_item:
                nbr_item_set.remove(int(tmp))
            nbr_item_set = np.array(list(nbr_item_set))
            counter_item_num = max(min(min(real_num//2, self.max_item - real_num), len(nbr_item_set)), 0)

            self.counter_item_num += counter_item_num

            if pos_num >= self.max_item:
                tmp = pos_item[:self.max_item]
                self.true_item_num += self.max_item
                self.pos_item_num += self.max_item
            else:
                if real_num >= self.max_item:
                    tmp = np.concatenate((pos_item, neg_item[:self.max_item - pos_num]))
                    self.true_item_num += self.max_item
                    self.pos_item_num += pos_num
                    self.neg_item_num += self.max_item - pos_num
                else:
                    self.true_item_num += real_num
                    self.pos_item_num += pos_num
                    self.neg_item_num += neg_num
                    if counter_item_num > 0:
                        tmp = np.concatenate((real_item, nbr_item_set[:counter_item_num]))
                    else:
                        tmp = real_item
                    if len(tmp) < self.max_item:
                        tmp = np.concatenate((tmp, np.zeros(self.max_item - len(tmp))))
                    if tmp.shape[-1] != self.max_item:
                        print(tmp.shape)
                        print("ELSE")
            self.available_items.append(tmp)

        t1 = time.time()
        print('rand time', t1 - t0)
        self.available_pos_num = torch.tensor(self.available_pos_num).long()

        self.available_items = torch.from_numpy(np.array(self.available_items)).long()
        self.available_items_mask = torch.ones_like(self.available_items).float()
        self.is_done = torch.zeros(self.user_num).bool()
        self.user_view_item = [set() for _ in range(self.user_num)]

        print("TRAIN user_num", self.user_num, 'true item num', self.true_item_num, 'pos item num ', self.pos_item_num, \
              'neg item num', self.neg_item_num, \
              "pos rate", self.pos_item_num / self.true_item_num, 'counter num', self.counter_item_num, \
              'ava item len', (self.true_item_num + self.counter_item_num) / self.user_num)

    def part_step(self, users, action_id, action, i, i_lo):
        self.memory[users, 0:-1] = self.memory[users, 1:].clone().detach()
        self.memory[users, -1] = action.squeeze(-1)

        time_id = int((self.ts[users, self.seq_id[users]] - self.Z_min_time) // self.Z_time_interval_size)
        if time_id < 0:
            time_id = 0
        counter_Z_u = self.Z_bin[time_id, int(action)]
        tmp_Z = (counter_Z_u / (self.Z_d[users, self.seq_id[users]]))
        if self.Z_d[users, self.seq_id[users]] == 0:
            gg = self.Z_bin_sum[time_id + 1]
            tmp_Z = self.Z_bin[time_id + 1, int(action)] / gg

        self.memory_Z[users, 0:-1] = self.memory_Z[users, 1:].clone().detach()
        self.memory_Z[users, -1] = tmp_Z.squeeze(-1)


        tmp = self.user_item_mat[tuple(i_lo)]
        if tmp != 0:
            if tmp > self.pos_line:
                tmp = 1
                self.user_view_item[i_lo[0]].add(action.item())
            else:
                tmp = 0
        else:
            tmp_beta = self.counterfactor.fin_beta[users].unsqueeze(0)
            re_p = self.counterfactor.train_inference(tmp_beta, self.memory_Z[users, -1].unsqueeze(0).unsqueeze(0),
                                                      self.memory[users, -1].unsqueeze(0).unsqueeze(0),
                                                      self.counterfactor.fin_U[users, self.seq_id[users], :].unsqueeze(
                                                          0).unsqueeze(0))
            re_p = re_p.squeeze()
            tmp = re_p
            if tmp >= self.check_line:
                tmp = 1
            else:
                tmp = 0
        self.seq_id[users] += 1
        self.memory_rate[users, 0:-1] = self.memory_rate[users, 1:].clone().detach()
        self.memory_rate[users, -1] = tmp
        tmp *= self.available_items_mask[users, action_id]
        self.available_items_mask[users, action_id] *= self.reward_decay
        return i, tmp

    def step(self, users, action_id, action):
        batch_size = users.shape[0]
        lo = torch.cat((users.unsqueeze(-1), action), dim=-1).tolist()
        rewards = []
        re_user = []
        res = []
        executor = ThreadPoolExecutor(max_workers=50)
        all_task = [executor.submit(self.part_step, users[i], action_id[i], action[i], i, i_lo) for i, i_lo in
                    enumerate(lo)]
        for data in as_completed(all_task):
            user_i, reward = data.result()
            if reward >= 0:
                rewards.append(reward)
                re_user.append(user_i)
        wait(all_task, return_when=ALL_COMPLETED)

        new_rewards = torch.zeros(users.shape)


        for i, i_user in enumerate(re_user):
            new_rewards[i_user] = rewards[i]

        rewards = new_rewards.clone()

        dones = torch.zeros(batch_size)
        dones = dones[re_user]

        return users, rewards, dones


class TestEnv(object):
    def __init__(self, user_num, memory, memory_rate, all_nbr, all_nbr_iid, all_nbr_iid_rate, args):
        print("test env")
        self.reward_decay = args.reward_decay
        self.env_reward_decay = args.reward_decay
        self.pos_line = args.pos_line
        self.user_num = user_num
        if memory != None:
            self.memory = memory.clone()
            self.memory_rate = memory_rate.clone()
        else:
            self.memory = None
            self.memory_rate = None

        self.user_len = [0]
        self.item_num = args.item_num
        self.all_nbr = all_nbr
        self.all_nbr_iid = all_nbr_iid
        self.all_nbr_iid_rate = all_nbr_iid_rate


        self.res_rate = torch.zeros((args.eval_batch_size, args.eval_max_item)).long()
        self.res_action = torch.zeros((args.eval_batch_size, args.eval_max_item)).long()
        self.res_action_id = torch.ones((args.eval_batch_size, args.eval_max_item)).long() * (-1)
        self.action_num = torch.tensor(0).long()
        self.ava_item = torch.randint(low=1, high=args.item_num, size=(args.eval_batch_size, args.item_num // 10))
        self.ava_item_mask = torch.ones((args.eval_batch_size, args.item_num // 10)).float()
        self.env_ava_item_mask = torch.ones((args.eval_batch_size, args.item_num // 10)).float()
        self.GT_user_emb = torch.from_numpy(np.load(f'../datasets/{args.dataset}/ground_truth_user_emb.npy'))#.to(args.device)
        self.GT_item_emb = torch.from_numpy(np.load(f'../datasets/{args.dataset}/ground_truth_item_emb.npy'))#.to(args.device)

    def reset(self, args, tmp_eval_batch_size):
        self.res_rate = torch.zeros((tmp_eval_batch_size, args.eval_max_item)).long()
        self.res_action = torch.zeros((tmp_eval_batch_size, args.eval_max_item)).long()
        self.res_action_id = torch.ones((tmp_eval_batch_size, args.eval_max_item)).long() * (-1)
        self.action_num = torch.tensor(0).long()
        self.ava_item = torch.randint(low=1, high=args.item_num, size=(tmp_eval_batch_size, args.item_num // 10))
        self.ava_item_mask = torch.ones((tmp_eval_batch_size, args.item_num // 10)).float()
        self.env_ava_item_mask = torch.ones((tmp_eval_batch_size, args.item_num // 10)).float()

    def step(self, users, action_id, action):
        users = users.squeeze().to('cpu')
        action_id = action_id.squeeze().to('cpu')
        action = action.squeeze().to('cpu')
        # print(users.device)
        # print(self.GT_user_emb.device)
        user_embedding = self.GT_user_emb[users]
        item_embedding = self.GT_item_emb[action].to('cpu')
        # print(user_embedding.device)
        # print(item_embedding.device)
        reward = torch.sum(user_embedding * item_embedding, dim=-1)
        tmp_env_ava_item_mask = self.env_ava_item_mask.gather(-1, action_id.unsqueeze(-1)).squeeze(-1)
        reward = (torch.sigmoid(reward)*tmp_env_ava_item_mask >= 0.5).long()
        self.res_action[:, self.action_num] = action
        self.res_action_id[:, self.action_num] = action_id
        self.res_rate[:, self.action_num] = reward
        self.ava_item_mask[range(len(users)), action_id] *= self.reward_decay
        self.env_ava_item_mask[range(len(users)), action_id] *= self.env_reward_decay
        self.action_num += 1
        if self.memory != None:
            self.memory[users, :-1] = self.memory[users, 1:].clone().detach()
            self.memory[users, -1] = action.clone()
            self.memory_rate[users, :-1] = self.memory_rate[users, 1:].clone().detach()
            self.memory_rate[users, -1] = reward.clone()