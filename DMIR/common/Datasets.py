from torch.utils.data import Dataset
import numpy as np
import torch

class TrainDataset(Dataset):
    def __init__(self, data, item_num):
        self.uid = data['train_uid']
        self.item_num = item_num

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]

        return user

class newEvalDataset(Dataset):
    def __init__(self, data, item_num):
        self.uid = data['eval_users']
        self.item_num = item_num

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]

        return user
class EvalDataset(Dataset):
    def __init__(self, data, item_num, is_test, max_pos):
        self.uid = data['train_uid']

        self.user_train = data['user_train'][()]
        self.user_valid = data['user_valid'][()]
        self.user_test = data['user_test'][()]

        self.eval_users = set(data['eval_users'].tolist())

        self.item_num = item_num
        self.is_test = is_test
        self.max_pos = max_pos

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        eval_iids= self.get_neg_samples(user)
        num = self.get_pos_num(user)
        return user, eval_iids, num

    def get_pos_num(self, user):
        return min(self.max_pos, len(self.user_valid[user]))

    def get_neg_samples(self, user):
        neg_sample_size = 100
        if user not in self.eval_users:
            return np.zeros((self.max_pos, neg_sample_size), dtype=np.int64)
        else:
            if self.is_test: eval_iid = self.user_test[user][0][0]
            else: eval_iid = self.user_valid[user][0][0]
            neg_list = [eval_iid]

        rated_set = set(np.array(self.user_train[user])[:, 0].tolist())
        if len(self.user_valid[user]): rated_set.add(self.user_valid[user][0][0])
        if len(self.user_test[user]): rated_set.add(self.user_test[user][0][0])
        rated_set.add(0)

        while len(neg_list) < neg_sample_size:
            neg = np.random.randint(low=1, high=self.item_num)
            while neg in rated_set:
                neg = np.random.randint(low=1, high=self.item_num)
            neg_list.append(neg)

        samples = np.array(neg_list, dtype=np.int64)
        all_samples = np.zeros((self.max_pos, neg_sample_size), dtype=np.int64)
        self_pos = len(self.user_valid[user])
        num = min(self_pos, self.max_pos)
        for i in range(num):
            all_samples[i] = samples
            all_samples[i][0] = self.user_valid[user][i][0]

        return all_samples

class EvalDataset2(Dataset):
    def __init__(self, data, item_num, is_test):
        self.uid = data['train_uid']

        self.user_train = data['user_train'][()]
        self.user_valid = data['user_valid'][()]
        self.user_test = data['user_test'][()]

        self.eval_users = set(data['eval_users'].tolist())

        self.item_num = item_num
        self.is_test = is_test

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        eval_iids = self.get_neg_samples(user)
        return user, eval_iids

    def get_neg_samples(self, user):
        neg_sample_size = 100
        if user not in self.eval_users:
            return np.zeros((neg_sample_size), dtype=np.int64)
        else:
            if self.is_test: eval_iid = self.user_test[user][0]
            else: eval_iid = self.user_valid[user][0]
            neg_list = [eval_iid]

        rated_set = set(self.user_train[user][:, 0].tolist())
        if len(self.user_valid[user]): rated_set.add(self.user_valid[user][0])
        if len(self.user_test[user]): rated_set.add(self.user_test[user][0])
        rated_set.add(0)

        while len(neg_list) < neg_sample_size:
            neg = np.random.randint(low=1, high=self.item_num)
            while neg in rated_set:
                neg = np.random.randint(low=1, high=self.item_num)
            neg_list.append(neg)

        samples = np.array(neg_list, dtype=np.int64)

        return samples

class DQN_pre_train_dataset(Dataset):
    def __init__(self, data, pos_line):
        self.uid = data['train_uid']
        self.seq = data['train_seq']
        self.rate = data['train_rate']
        self.rate[self.rate <= pos_line] = 0
        self.rate[self.rate > pos_line] = 1

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        seq = self.seq[idx]
        rate = self.rate[idx]

        return user, seq, rate

class DQN_pre_eval_dataset(Dataset):
    def __init__(self, data, pos_line):
        self.test_u = data['train_uid']
        self.uid = data['eval_users']
        self.seq = data['train_seq']
        self.user_train = data['user_train'][()]
        self.all_valid = data['all_valid'][()]
        self.all_valid_per_num = data['all_valid_per_num'][()]
        self.rate = data['train_rate']
        self.rate[self.rate <= pos_line] = 0
        self.rate[self.rate > pos_line] = 1
        self.pos_line = pos_line

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        test_u = self.test_u[idx]
        seq = self.seq[user-1]
        rate = self.rate[user-1]
        print("idx",idx,"test u",test_u,'user',user)
        return user, seq, rate, self.all_valid[user][:, 0], self.all_valid[user][:, 2]>self.pos_line

class CAR_pre_TrainDataset(Dataset):
    def __init__(self, data, pos_line):
        self.uid = data['train_uid']
        self.seq = data['train_seq']
        self.rate = data['train_rate']
        self.rate[self.rate <= pos_line] = 0
        self.rate[self.rate > pos_line] = 1
        self.nbr = data['train_nbr']
        self.nbr_iid = data['train_nbr_iid']
        self.nbr_iid_rate = data['train_nbr_iid_rate']
        self.nbr_iid_rate[self.nbr_iid_rate <= pos_line] = 0
        self.nbr_iid_rate[self.nbr_iid_rate > pos_line] = 1


        self.user_train = data['user_train'][()]
        self.all_valid = data['all_valid'][()]

        self.eval_users = set(data['eval_users'].tolist())

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        seq = self.seq[idx]
        rate = self.rate[idx]
        nbr = self.nbr[idx]
        nbr_iid = self.nbr_iid[idx]
        nbr_iid_rate = self.nbr_iid_rate[idx]

        return user, seq, rate, nbr, nbr_iid, nbr_iid_rate

class CAR_pre_EvalDataset(Dataset):
    def __init__(self, data, pos_line):
        self.uid = data['eval_users']
        self.user_train = data['user_train'][()]
        self.all_valid = data['all_valid'][()]
        self.all_valid_per_num = data['all_valid_per_num'][()]

        self.seq = data['train_seq']
        self.rate = data['train_rate']
        self.rate[self.rate <= pos_line] = 0
        self.rate[self.rate > pos_line] = 1
        self.nbr = data['train_nbr']
        self.nbr_iid = data['train_nbr_iid']
        self.nbr_iid_rate = data['train_nbr_iid_rate']
        self.nbr_iid_rate[self.nbr_iid_rate <= pos_line] = 0
        self.nbr_iid_rate[self.nbr_iid_rate > pos_line] = 1
        self.pos_line = pos_line

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        seq = self.seq[user-1]
        rate = self.rate[user-1]
        nbr = self.nbr[user-1]
        nbr_iid = self.nbr_iid[user-1]
        nbr_iid_rate = self.nbr_iid_rate[user-1]
        return user, seq, rate, nbr, nbr_iid, nbr_iid_rate, self.all_valid[user][:,0], self.all_valid[user][:,2]>self.pos_line


class Light_GCN_TrainDataset(Dataset):
    def __init__(self, data, pos_line):
        self.uid = data['train_uid']
        self.seq = data['train_seq']

        self.rate = data['train_rate']
        self.rate[self.rate <= pos_line] = 0
        self.rate[self.rate > pos_line] = 1
        self.pos = self.seq.copy()
        self.pos[self.rate<1] = 0
        self.neg = self.seq.copy()
        self.neg[self.rate>0] = 0

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        pos = self.pos[idx]
        neg = self.neg[idx]

        return user, pos, neg

class BPRMF_TrainDataset(Dataset):
    def __init__(self, data, pos_line):
        self.uid = data['train_uid']
        self.seq = data['train_seq']


        self.rate = data['train_rate']
        self.rate[self.rate <= pos_line] = 0
        self.rate[self.rate > pos_line] = 1
        self.pos = self.seq.copy()
        self.pos[self.rate<1] = 0
        self.pos = self.pos[:,1:]
        self.neg = self.seq.copy()
        self.neg[self.rate>0] = 0
        self.neg = self.neg[:,1:]
        self.seq = self.seq[:, :-1]

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        seq = self.seq[idx]
        pos = self.pos[idx]
        neg = self.neg[idx]

        return user, seq, pos, neg