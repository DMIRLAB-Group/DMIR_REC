import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import logging
import sys
import pickle as pkl

class TrainDataset(Dataset):
    def __init__(self, data, item_num, is_train, is_test, pos_line):
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
        self.Z = data['train_Z']
        self.user_train = data['user_train'][()]
        self.all_valid = data['all_valid'][()]
        self.eval_users = set(data['eval_users'].tolist())
        self.item_num = item_num
        self.is_train = is_train
        self.is_test = is_test

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        seq = self.seq[idx]
        rate = self.rate[idx]
        nbr = self.nbr[idx]
        nbr_iid = self.nbr_iid[idx]
        nbr_iid_rate = self.nbr_iid_rate[idx]
        Z = self.Z[idx]

        return user, seq, rate, nbr, nbr_iid, nbr_iid_rate, Z

class EvalDataset(Dataset):
    def __init__(self, data, pos_line):
        self.uid = data['eval_users']
        self.user_train = data['user_train'][()]
        self.all_valid = data['all_valid'][()]
        self.all_valid_per_num = data['all_valid_per_num'][()]

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        return user, self.all_valid[user][:,0], self.all_valid[user][:,2], self.all_valid[user][:,3], self.all_valid_per_num[user]

class DC_EvalDataset(Dataset):
    def __init__(self, data, pos_line):
        self.uid = data['eval_users']
        self.user_train = data['user_train'][()]
        self.all_valid = data['all_valid'][()]
        self.all_valid_per_num = data['all_valid_per_num'][()]

        self.seq = data['train_seq']
        self.rate = data['train_rate']
        self.rate[self.rate <= pos_line] = 0
        self.rate[self.rate > pos_line] = 1

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        return user, self.seq[user-1], self.rate[user-1], self.all_valid[user][:,0], self.all_valid[user][:,2], \
               self.all_valid[user][:,3], self.all_valid_per_num[user]

def load_ds(args, item_num):
    data = np.load(f'../datasets/{args.dataset}/{args.use_all_Z}Z_{args.time_interval_size_day}day_{args.valid_scale}valid_scale_{args.seq_len}seq_len_processed_data.npz',allow_pickle=True)

    train_loader = DataLoader(
        dataset=TrainDataset(data, item_num, True, False, args.pos_line),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False)

    val_loader = DataLoader(
        dataset=EvalDataset(data, args.pos_line),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False)

    user_train = data['user_train'][()]
    eval_users = data['eval_users']

    return train_loader, val_loader, user_train, eval_users

def save_pkl(file, obj):
    with open(file, 'wb') as f:
        pkl.dump(obj, f)


def load_pkl(file):
    with open(file, 'rb') as f:
        data = pkl.load(f)
    return data


def parse_sampled_batch(batch):
    user, seq, rate, nbr, nbr_iid, nbr_iid_rate, Z = batch
    user = user.long()
    seq = seq.long()
    rate = rate.long()
    nbr = nbr.long()
    nbr_iid = nbr_iid.long()
    nbr_iid_rate = nbr_iid_rate.long()
    tmp1 = torch.where(seq>0)
    tmp2 = torch.where(rate>0)
    pos_num = len(tmp2[0])
    neg_num = len(tmp1[0])-pos_num
    Z = Z.float()
    batch = [user, seq, rate, nbr, nbr_iid, nbr_iid_rate, Z]
    indices = torch.where(seq != 0)
    return batch, indices, pos_num, neg_num

def get_logger(filename=None):

    log_formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)-5.5s] %(message)s', datefmt='%Y-%b-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.INFO)
    logger.addHandler(std_handler)

    return logger
