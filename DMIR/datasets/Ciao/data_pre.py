
import json
import pandas as pd
import scipy.sparse as sp
import numpy as np
import argparse

import os
import re

from collections import defaultdict

from tqdm import tqdm
import gc
import networkx as nx

from time import time, mktime, strptime
import joblib
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat

np.set_printoptions(threshold=1000000)

saved_path = './'


def preprocess_uir(df, prepro='origin', binary=False, pos_threshold=None, level='ui'):

    # set rating >= threshold as positive samples
    if pos_threshold is not None:
        df = df.query(f'rate >= {pos_threshold}').reset_index(drop=True)

    # reset rating to interaction, here just treat all rating as 1
    if binary:
        df['rate'] = 1.0

    # which type of pre-dataset will use
    if prepro == 'origin':
        pass

    elif prepro.endswith('filter'):
        pattern = re.compile(r'\d+')
        filter_num = int(pattern.findall(prepro)[0])

        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])

        if level == 'ui':
            df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'u':
            df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'i':
            df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()
        else:
            raise ValueError(f'Invalid level value: {level}')

        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()



    elif prepro.endswith('core'):
        pattern = re.compile(r'\d+')
        core_num = int(pattern.findall(prepro)[0])

        def filter_user(df):
            tmp = df.groupby(['user'], as_index=False)['item'].count()
            tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=['user'])
            df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)

            return df

        def filter_item(df):
            tmp = df.groupby(['item'], as_index=False)['user'].count()
            tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=['item'])
            df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)

            return df

        if level == 'ui':
            while 1:
                df = filter_user(df)
                df = filter_item(df)
                chk_u = df.groupby('user')['item'].count()
                chk_i = df.groupby('item')['user'].count()
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif level == 'u':
            df = filter_user(df)
        elif level == 'i':
            df = filter_item(df)
        else:
            raise ValueError(f'Invalid level value: {level}')

        gc.collect()

    else:
        raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')

    return df


def save_pkl(file, obj, compress=0):
    joblib.dump(value=obj, filename=file, compress=compress)


def load_pkl(file):
    return joblib.load(file)

def filter_and_reid():
    save_path = saved_path + 'datasets/Yelp/reid_u2ui.npz'
    if os.path.exists(save_path):
        return
    u2ui = np.load(saved_path + f'datasets/Yelp/u2ui.npz')
    u2u, u2i = u2ui['u2u'], u2ui['u2i']
    df = pd.DataFrame(data=u2i, columns=['user', 'item', 'ts', 'rate'])
    df.drop_duplicates(subset=['user', 'item', 'ts', 'rate'], keep='first', inplace=True)

    print('Raw u2i', df.shape)
    print('min user =', df['user'].min())
    print('max user =', df['user'].max())
    print('num user =', len(np.unique(df.values[:, 0])))
    print('min item =', df['item'].min())
    print('max item =', df['item'].max())
    print('num item =', len(np.unique(df.values[:, 1])))

    df = preprocess_uir(df, prepro='3filter', level='u')

    print('Processed u2i', df.shape)
    print('min user =', df['user'].min())
    print('max user =', df['user'].max())
    print('num user =', len(np.unique(df.values[:, 0])))
    print('min item =', df['item'].min())
    print('max item =', df['item'].max())
    print('num item =', len(np.unique(df.values[:, 1])))

    df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
    u2i = df.values

    user_idmap = defaultdict(int)  # src id -> new id
    user_idmap[0] = 0
    user_num = 1

    for i, (user, item, ts, rate) in tqdm(enumerate(u2i)):
        if user_idmap[user] == 0:
            user_idmap[user] = user_num
            user_num += 1

        u2i[i, 0] = user_idmap[user]

    print('Raw u2u edges:', len(u2u))
    new_uu_elist = []
    for u1, u2 in tqdm(u2u):
        new_u1 = user_idmap[u1]
        new_u2 = user_idmap[u2]
        if new_u1 and new_u2:
            new_uu_elist.append([new_u1, new_u2])

    print('Processed u2u edges:', len(new_uu_elist))
    u2u = np.array(new_uu_elist).astype(np.int32)
    u2i = u2i.astype(np.int32)

    np.savez(file=save_path, u2u=u2u, u2i=u2i)

    print('saved at', save_path)


def delete_isolated_user(u2i, uu_elist):
    save_path = saved_path + 'noiso_reid_u2ui.npz'


    print('Building u2u graph...') # user to user graph
    user_num = np.max(u2i[:, 0]) + 1
    g = nx.Graph()
    g.add_nodes_from(list(range(user_num)))
    g.add_edges_from(uu_elist)
    g.remove_node(0)

    isolated_user_set = set(nx.isolates(g))
    print('Isolated user =', len(isolated_user_set))

    new_u2i = []
    for user, item, ts, rate in tqdm(u2i):
        if user not in isolated_user_set:
            new_u2i.append([user, item, ts, rate])

    new_u2i = np.array(new_u2i, dtype=np.int32)

    print('No isolated user u2i =', new_u2i.shape)

    user_idmap = defaultdict(int)  # src id -> new id
    user_idmap[0] = 0
    user_num = 1
    for i, (user, item, ts, rate) in tqdm(enumerate(new_u2i)):
        if user_idmap[user] == 0:
            user_idmap[user] = user_num
            user_num += 1

        new_u2i[i, 0] = user_idmap[user]

    new_uu_elist = []
    for u1, u2 in tqdm(uu_elist):
        new_u1 = user_idmap[u1]
        new_u2 = user_idmap[u2]
        if new_u1 and new_u2:
            new_uu_elist.append([new_u1, new_u2])

    new_uu_elist = np.array(new_uu_elist, dtype=np.int32)

    df = pd.DataFrame(data=new_u2i, columns=['user', 'item', 'ts', 'rate'])
    df['item'] = pd.Categorical(df['item']).codes + 1

    user_num = df['user'].max() + 1 # 2343
    item_num = df['item'].max() + 1 # 77541

    print('min user =', df['user'].min())
    print('max user =', df['user'].max())
    num_user = len(np.unique(df.values[:, 0]))
    print('num user =', num_user)

    print('min item =', df['item'].min())
    print('max item =', df['item'].max())
    num_item = len(np.unique(df.values[:, 1]))
    print('num item =', num_item)

    print(f'Loaded Yelp dataset with {user_num} users, {item_num} items, '
          f'{len(df.values)} u2i, {len(new_uu_elist)} u2u. ')

    new_u2i = df.values.astype(np.int32)

    np.savez(file=save_path, u2u=new_uu_elist, u2i=new_u2i)
    np.save(saved_path + 'user_item_num.npy', np.array([num_user, num_item]))
    return num_user, num_item


def data_partition(df):
    print('Splitting train/val/test set...')
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    eval_users = []

    train_mat = defaultdict(int)

    user_items_dict = defaultdict(list)

    def apply_fn1(grp):
        key_id = grp['user'].values[0]
        user_items_dict[key_id] = grp[['item', 'ts', 'rate']].values

    df.groupby('user').apply(apply_fn1)

    print('Groupby user finished.')
    item_nums_per_user = []
    for user in tqdm(user_items_dict.keys()):
        nfeedback = len(np.array(user_items_dict[user])[:,2])
        item_nums_per_user.append(nfeedback)
        if nfeedback < 20:
            user_train[user] = user_items_dict[user]
            for i in range(user_items_dict[user].shape[0]):
                train_mat[user, user_items_dict[user][i, 0]] = user_items_dict[user][i, -1]
        else:
            valid_num = max(1, math.floor(valid_scale * user_items_dict[user].shape[0]))
            eval_users.append(user)
            user_train[user] = user_items_dict[user][:-valid_num]
            for i in range(user_items_dict[user].shape[0]-valid_num):
                train_mat[user, user_items_dict[user][i, 0]] = user_items_dict[user][i, -1]

            valid_item = user_items_dict[user][-valid_num:]
            user_valid[user] = valid_item

    return user_train, user_valid, eval_users, item_nums_per_user, train_mat

def load_ds(dataset='Ciao'):

    rating = pd.DataFrame()

    rating_mat = loadmat(f'rating_with_timestamp.mat')
    if dataset == 'Ciao':
        rating = rating_mat['rating'] # ndarry:(284086,6)
    elif dataset == 'Epinions':
        rating = rating_mat['rating_with_timestamp']
    print(rating[:10])
    df = pd.DataFrame(data=rating, columns=['user', 'item', 'cate', 'rate', 'help', 'ts'])
    print('data')
    print(df[:10])
    df.drop(columns=['cate', 'help'], inplace=True)

    df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
    df = preprocess_uir(df, prepro='origin')
    df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
    df = df.loc[:,['user','item','ts','rate']]

    u2i = df.to_numpy()


    print("df max",df.max())
    print('df min',df.min())
    print("dif user id",len(dict(df['user'].value_counts())))
    print('dif item id',len(dict(df['item'].value_counts())))
    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1
    print('user num',user_num,'item num',item_num)
    uu_elist = loadmat(f'trust.mat')['trust']

    num_user, num_item = delete_isolated_user(u2i,uu_elist)

    return num_user, num_item

def gen_and_save_u2u_dict_and_split(num_user, num_item):

    u2ui = np.load(saved_path + 'noiso_reid_u2ui.npz')

    print('Building u2u graph...')
    g = nx.Graph()
    g.add_nodes_from(list(range(num_user)))
    g.add_edges_from(u2ui['u2u'])
    g.remove_node(0)

    print('To undirected graph...')
    g.to_undirected()
    u2u_dict = nx.to_dict_of_lists(g)

    df = pd.DataFrame(data=u2ui['u2i'], columns=['user', 'item', 'ts', 'rate'])
    print('Raw u2i =', df.shape) # DataFrame:(146777, 4)
    df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True) # DataFrame:(146777, 4)
    df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
    print('Processed u2i =', df.shape) # Processed u2i = (146777, 4)


    user_train, user_valid, eval_users, item_nums_per_user, train_mat = data_partition(df)
    save_path = saved_path + f'{valid_scale}u2u_split_dicts.pkl'
    save_pkl(save_path, [
        u2u_dict, user_train, user_valid,
        eval_users, item_nums_per_user, train_mat])

    item_nums_per_user.sort()
    item_nums_per_user = np.array(item_nums_per_user) # ndarrsy:(2342,)
    print("item_nums_per_user mean", item_nums_per_user.mean(), 'max', item_nums_per_user.max(), 'min',
          item_nums_per_user.min(), 'mid',
          item_nums_per_user[item_nums_per_user.shape[0] // 2]) # item_nums_per_user mean 62.67164816396242 max 1543 min 5 mid 30
    print('saved at', save_path)


def get_nbr(u2u, user, nbr_maxlen):
    nbr = np.zeros([nbr_maxlen, ], dtype=np.int32)
    nbr_len = len(u2u[user])
    if nbr_len == 0:
        pass
    elif nbr_len > nbr_maxlen:
        np.random.shuffle(u2u[user])
        nbr[:] = u2u[user][:nbr_maxlen]
    else:
        nbr[:nbr_len] = u2u[user]

    return nbr


def get_nbr_iids(user_train, user, nbrs, time_splits):
    nbr_maxlen = len(nbrs)
    seq_maxlen = len(time_splits)

    nbrs_iids = np.zeros((nbr_maxlen, seq_maxlen), dtype=np.int32)
    nbrs_iids_rate = np.zeros((nbr_maxlen, seq_maxlen), dtype=np.int32)

    start_idx = np.nonzero(time_splits)[0]
    if len(start_idx) == 0:
        return nbrs_iids, nbrs_iids_rate
    else:
        start_idx = start_idx[0]

    user_first_ts = time_splits[start_idx]
    user_last_ts = time_splits[-1]

    for i, nbr in enumerate(nbrs):
        if nbr == 0 or nbr == user:
            continue

        nbr_hist = user_train[nbr]

        if len(nbr_hist) == 0:
            continue

        nbr_first_ts = nbr_hist[0][1]
        nbr_last_ts = nbr_hist[-1][1]

        if nbr_first_ts > user_last_ts or nbr_last_ts <= user_first_ts:
            continue

        sample_list = list()

        tmp_all_nbr_iid_list = list()
        la = 0
        for j in range(start_idx, seq_maxlen):
            if j != 0:
                start_time = time_splits[j - 1]
            else:
                start_time = -1
            end_time = time_splits[j]

            if start_time != end_time:

                sample_list = list(filter(None, map(
                    lambda x: (x[0], x[2]) if x[1] > start_time and x[1] <= end_time else None, nbr_hist
                )))

            if len(sample_list):
                tmp = np.random.choice(range(len(sample_list)))
                nbrs_iids[i, j] = sample_list[tmp][0]
                nbrs_iids_rate[i, j] = sample_list[tmp][1]

    return nbrs_iids, nbrs_iids_rate

def get_Z_bin(item_num):
    u2u_dict, user_train, user_valid, eval_users, item_nums_per_user, train_mat = \
        load_pkl(saved_path + f'{valid_scale}u2u_split_dicts.pkl')
    da = np.load(saved_path + 'noiso_reid_u2ui.npz')

    u2i = da['u2i']
    print("check item time")
    print("u2i len", len(u2i))
    ti = u2i[:, 2]
    time_len = ti.max() - ti.min()
    min_time = ti.min()
    time_interval_size = 3600*24*time_interval_size_day

    time_interval_num = time_len // time_interval_size + 2

    train_Z_bin = np.zeros((time_interval_num, item_num))
    all_Z_bin = np.zeros((time_interval_num, item_num))
    # time and item in train
    for user in user_train:
        for i in user_train[user]:
            interval_idx = (i[1] - min_time) // time_interval_size + 1
            train_Z_bin[interval_idx, i[0]] += 1
            all_Z_bin[interval_idx, i[0]] += 1


    for user in user_valid:
        for i in user_valid[user]:
            interval_idx = (i[1] - min_time) // time_interval_size + 1
            all_Z_bin[interval_idx, i[0]] += 1
    sum_train_Z = train_Z_bin.sum(-1)
    sum_all_Z = all_Z_bin.sum(-1)
    print('interval_num',time_interval_num,'train zero',len(sum_train_Z==0),'all zero',len(sum_all_Z))
    np.savez(saved_path + f'{time_interval_size_day}day_{valid_scale}valid_scale_Z_bin.npz', train_Z_bin=train_Z_bin, all_Z_bin=all_Z_bin, time_interval_size=time_interval_size,
             min_time=ti.min(), max_time=ti.max())

def all_get_Z(user, seq, ts, nbrs, train_Z_bin, all_Z_bin, time_interval_size, user_train, user_valid, min_time):
    Z = np.zeros(seq_maxlen, dtype=np.float32)
    Z_d = np.zeros(seq_maxlen, dtype=np.float32)
    train_Z_d = train_Z_bin.sum(-1)
    all_Z_d = all_Z_bin.sum(-1)
    time_interval_num = train_Z_bin.shape[0]
    valid_Z = []
    for i in range(seq_maxlen):
        if seq[i] == 0:
            continue
        time_id = (ts[i] - min_time) // time_interval_size
        if train_Z_d[time_id] == 0:
            train_Z_d[time_id] = 1
        Z[i] = float(train_Z_bin[time_id, seq[i]]) / float(train_Z_d[time_id])
        Z_d[i] = float(train_Z_d[time_id])

    if user_valid[0, 0] != -1:
        for (item, ts, ra) in user_valid:
            time_id = (ts - min_time) // time_interval_size
            if all_Z_d[time_id] == 0:
                all_Z_d[time_id] = 1
            valid_Z.append(float(all_Z_bin[time_id, item]) / all_Z_d[time_id])
        valid_Z = np.array(valid_Z)

        return Z, valid_Z, Z_d
    else:
        return Z, np.array([-1]), Z_d

def nbr_get_Z(user, seq, ts, nbrs, train_Z_bin, all_Z_bin, time_interval_size, user_train, user_valid, min_time):
    Z = np.zeros(seq_maxlen, dtype=np.float32)
    time_interval_num = train_Z_bin.shape[0]
    Z_d_sets = [set() for _ in range(train_Z_bin.shape[0])]
    time_ids = set()

    for i in range(seq_maxlen):
        if ts[i] == 0:
            continue
        time_ids.add((ts[i] - min_time) // time_interval_size)
    if user_valid[0,0] != -1:
        for i in range(len(user_valid)):
            time_ids.add((user_valid[i, 1] - min_time) // time_interval_size)


    for nbr in nbrs:
        if nbr == 0 or nbr == user:
            continue
        for x in user_train[nbr]:
            time_id = (x[1] - min_time) // time_interval_size
            if time_id not in time_ids:
                continue
            Z_d_sets[time_id].add(x[0])

    for i in range(seq_maxlen):
        if seq[i] == 0:
            continue
        time_id = (ts[i] - min_time) // time_interval_size
        Z_d_sets[time_id].add(seq[i])


    Z_d = np.zeros(seq_maxlen,dtype=np.float32)
    time_id_map = {}
    for i in range(seq_maxlen):
        if seq[i] == 0:
            continue
        time_id = (ts[i] - min_time) // time_interval_size
        time_id_map[time_id] = i
        for time_nbr in Z_d_sets[time_id]:
            Z_d[i] += train_Z_bin[time_id, time_nbr]

    for i in range(seq_maxlen):
        if seq[i] == 0:
            continue
        time_id = (ts[i] - min_time) // time_interval_size
        Z[i] = float(train_Z_bin[time_id, seq[i]]) / float(Z_d[i])

    if user_valid[0, 0] != -1:
        valid_Z_d = defaultdict(float)
        flag = defaultdict(int)
        valid_Z = []
        for (item, ts, ra) in user_valid:
            time_id = (ts - min_time) // time_interval_size
            valid_Z_d[time_id] += all_Z_bin[time_id, item]
            if flag[time_id] == 1:
                continue
            flag[time_id] = 1
            for time_nbr in Z_d_sets[time_id]:
                valid_Z_d[time_id] += all_Z_bin[time_id, time_nbr]


        for (item, ts, ra) in user_valid:
            time_id = (ts - min_time) // time_interval_size
            valid_Z.append(float(all_Z_bin[time_id, item])/valid_Z_d[time_id])
        valid_Z = np.array(valid_Z)

        return Z, valid_Z, Z_d
    else:
        return Z, np.array([-1]), Z_d


def get_self_nbr_item_set(user_num):
    u2u_dict, user_train, user_valid,eval_users, item_nums_per_user, train_mat = \
        load_pkl(saved_path + f'{valid_scale}u2u_split_dicts.pkl')
    self_nbr_item_sets = [set()]
    for user in range(1, user_num):
        tmp = set()
        for nbr in u2u_dict[user]:
            for i in user_train[nbr]:
                tmp.add(i[0])
        for i in user_train[user]:
           tmp.add(i[0])
        self_nbr_item_sets.append(tmp)
    np.save(saved_path + f'{valid_scale}self_nbr_item_set.npy', self_nbr_item_sets)


def gen_and_save_all_user_batches(user_num, item_num):
    t0 = time()
    u2u_dict, user_train, user_valid, eval_users, item_nums_per_user, train_mat = \
        load_pkl(saved_path + f'{valid_scale}u2u_split_dicts.pkl')
    train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.int32)

    dict.update(train_matrix, train_mat)
    tmp = np.load(saved_path + f'{time_interval_size_day}day_{valid_scale}valid_scale_Z_bin.npz')
    time_interval_size = tmp['time_interval_size']
    train_Z_bin = tmp['train_Z_bin']
    all_Z_bin = tmp['all_Z_bin']
    min_time = tmp['min_time']
    t1 = time()
    print('load time', t1 - t0)

    def sample_one_user(user):
        print("start user", user)
        seq = np.zeros(seq_maxlen, dtype=np.int32)
        rate = np.zeros(seq_maxlen, dtype=np.int32)
        ts = np.zeros(seq_maxlen, dtype=np.int32)
        idx = seq_maxlen - 1
        for (item, time_stamp, ra) in reversed(user_train[user]):
            seq[idx] = item
            rate[idx] = ra
            ts[idx] = time_stamp
            idx -= 1
            if idx == 0:
                break

        nbr = get_nbr(u2u_dict, user, nbr_maxlen)
        nbr_iid, nbr_iid_rate = get_nbr_iids(user_train, user, nbr, ts)
        if len(user_valid[user]) > 0:
            if use_all_Z == 1:
                Z, valid_Z, Z_d = all_get_Z(user, seq, ts, nbr, train_Z_bin, all_Z_bin, time_interval_size, user_train, user_valid[user], min_time)
            else:
                Z, valid_Z, Z_d = nbr_get_Z(user, seq, ts, nbr, train_Z_bin, all_Z_bin, time_interval_size, user_train,
                                            user_valid[user], min_time)
        else:
            if use_all_Z == 1:
                Z, valid_Z, Z_d = all_get_Z(user, seq, ts, nbr, train_Z_bin, all_Z_bin, time_interval_size, user_train, np.array([[-1]]), min_time)
            else:
                Z, valid_Z, Z_d = nbr_get_Z(user, seq, ts, nbr, train_Z_bin, all_Z_bin, time_interval_size, user_train,
                                            np.array([[-1]]), min_time)
        return user, seq, rate, ts, nbr, nbr_iid, nbr_iid_rate, Z, valid_Z, Z_d

    uid_list = []
    seq_list = []
    rate_list = []
    nbr_list = []
    nbr_iid_list = []
    nbr_iid_rate_list = []
    Z_list = []
    all_valid = defaultdict(list)
    all_valid_per_num = defaultdict(int)
    Z_d_list = []
    ts_list = []
    for user in tqdm(range(1, user_num)):
        user, seq, rate, ts, nbr, nbr_iid, nbr_iid_rate, Z, valid_Z, Z_d = sample_one_user(user)
        uid_list.append(user)
        seq_list.append(seq)
        rate_list.append(rate)
        nbr_list.append(nbr)
        nbr_iid_list.append(nbr_iid)
        nbr_iid_rate_list.append(nbr_iid_rate)
        Z_list.append(Z)

        if valid_Z[0] != -1:
            all_valid[user] = np.concatenate((user_valid[user], np.expand_dims(valid_Z, -1)), axis=-1)
            l = len(all_valid[user])
            if l >= max_per_user_valid:
                all_valid_per_num[user] = max_per_user_valid
                all_valid[user] = all_valid[user][:max_per_user_valid, :]
            else:
                all_valid_per_num[user] = l
                all_valid[user] = np.concatenate((all_valid[user], np.zeros((max_per_user_valid - l, 4))), axis=0)

        Z_d_list.append(Z_d)
        ts_list.append(ts)

    np.savez(
        saved_path + f'{use_all_Z}Z_{time_interval_size_day}day_{valid_scale}valid_scale_{seq_maxlen}seq_len_processed_data.npz',
        user_train=user_train,
        train_matrix=train_matrix,
        eval_users=np.array(eval_users, dtype=np.int32),
        train_uid=np.array(uid_list, dtype=np.int32),
        train_seq=np.array(seq_list, dtype=np.int32),
        train_rate=np.array(rate_list, dtype=np.int32),
        train_nbr=np.array(nbr_list, dtype=np.int32),
        train_nbr_iid=np.array(nbr_iid_list),
        train_nbr_iid_rate=np.array(nbr_iid_rate_list),
        train_Z=np.array(Z_list, dtype=np.float32),
        all_valid=all_valid,
        all_valid_per_num=all_valid_per_num,
        item_nums_per_user=np.array(item_nums_per_user, dtype=np.int32),
        train_Z_d=np.array(Z_d_list, dtype=np.float32),
        train_ts=np.array(ts_list),
        u2u_dict=u2u_dict
    )

    print(f'saved at {valid_scale}processed_data.npz')





def preprocess():

    user_num, item_num = load_ds(dataset)
    user_num += 1
    item_num += 1
    print("num_user", user_num)
    print('num_item', item_num)
    gen_and_save_u2u_dict_and_split(user_num, item_num)
    get_Z_bin(item_num)
    get_self_nbr_item_set(user_num)
    print('fin self_nbr_item_set',valid_scale)
    gen_and_save_all_user_batches(user_num, item_num)
    print("fin",valid_scale)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='Ciao')
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_maxlen', type=int, default=20)
    parser.add_argument('--nbr_maxlen', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--valid_scale', type=float, default=0.5)
    parser.add_argument('--max_per_user_valid', type=int, default=50)
    parser.add_argument('--time_interval_size_day', type=int, default=30)
    parser.add_argument('--use_all_Z', type=int, default=1)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    dataset = args.dataset
    seq_maxlen = args.seq_maxlen
    nbr_maxlen = args.nbr_maxlen
    valid_scale = args.valid_scale
    max_per_user_valid = args.max_per_user_valid
    time_interval_size_day = args.time_interval_size_day
    use_all_Z = args.use_all_Z
    t0 = time()
    preprocess()
    t1 = time()
    print('all time',t1-t0)
