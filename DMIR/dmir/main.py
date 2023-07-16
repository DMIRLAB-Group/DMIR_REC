import sys, os
sys.path.append('../')
import torch
import datetime
import numpy as np
import argparse
from torch.utils.data import DataLoader
import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import time

from common.utils import make_dir, get_logger
from common.Datasets import TrainDataset
from common.Datasets import newEvalDataset
from agent import DMIR
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

def train(args, train_loader, eval_loader, env, agent, writer, eval_users, last_step, user_valid, logger, norm_i_embedding, curr_time, CA_opt=None):
    step = last_step

    for i_batch, batch in enumerate(train_loader):

        users = batch
        t0 = time()

        users = users.long()
        batch_ava_items = env.available_items[users].clone().to(args.device)
        batch_nbr = env.all_nbr[users].clone().to(args.device)
        batch_nbr_iid = env.all_nbr_iid[users].clone().to(args.device)
        batch_nbr_iid_rate = env.all_nbr_iid_rate[users].clone().to(args.device)

        st_users = users.clone()
        agent.frame_idx = 0
        for iner_step in range(args.seq_len):


            ch_time_st = time()
            tmp_memory = env.memory[users].clone()
            tmp_memory_rate = env.memory_rate[users].clone()
            action_id, action = agent.choose_action(users.to(args.device), env.memory[users].to(args.device),
                                                    env.memory_rate[users].to(args.device), batch_nbr, batch_nbr_iid,
                                                    batch_nbr_iid_rate,
                                                    batch_ava_items,
                                                    env.available_items_mask[users].to(args.device), env.is_done[users])

            users, reward, done = env.step(users, action_id, action)
            if iner_step == args.seq_len-1:
                done = torch.zeros(done.shape)

            for i in range(users.shape[0]):
                agent.buffer.push(users[i], tmp_memory[i], tmp_memory_rate[i], action[i].clone(), reward[i].clone(),
                                  env.memory[users[i]].clone(), env.memory_rate[users[i]].clone(), done[i].clone(),
                                  env.available_items_mask[users[i]].clone())

            if len(agent.buffer.buffer) >= args.update_size:
                agent.policy_net.train()
                agent.target_net.train()
                loss = agent.update(env.available_items, env.all_nbr, env.all_nbr_iid, env.all_nbr_iid_rate, args)
                writer.add_scalar('\\Train/loss\\', loss, step)
                writer.flush()
            step += 1
            agent.frame_idx+=1
            if step % args.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())


        if CA_opt != None:
            env.counterfactor.train()
            CA_opt.zero_grad()
            ca_rate = env.memory_rate[st_users]
            ca_loss = env.counterfactor(
                (st_users, env.memory[st_users].detach(), ca_rate.detach(), env.all_nbr[st_users].detach(),
                 env.all_nbr_iid[st_users].detach(), env.all_nbr_iid_rate[st_users].detach(),
                 env.memory_Z[st_users].detach()))
            ca_loss.backward()
            CA_opt.step()
            env.counterfactor.eval()
            writer.add_scalar('\\Train/CA_loss\\', ca_loss, step)
            torch.save(env.counterfactor.state_dict(),
                      f'../envs/saved_models/{args.dataset}/tmp/{args.model}{curr_time}updated_CA.pth')
        t1 = time()
        logger.info(f'batch {i_batch} time {t1 - t0}')
    agent.lr_scheduler.step()
    return env, step


def eval(args, loader, test_env, agent, writer, last_step, eval_users, norm_i_embedding, K, logger, ty='Valid', pre=False):
    agent.policy_net.eval()
    agent.target_net.eval()
    def get_entropy(i, j):
        tmp = torch.cat((norm_i_embedding[i].unsqueeze(-2), norm_i_embedding[j].unsqueeze(-2)), dim=-2).to(args.device)
        means = tmp.mean(dim=-1)
        stds = tmp.std(dim=-1)
        KL = torch.log(torch.sqrt(stds[:, 1]) / torch.sqrt(stds[:, 0])) - 1.0 / 2.0 + \
             (stds[:, 0] + torch.square(means[:, 0] - means[:, 1])) / (2 * stds[:, 1])
        return KL

    with torch.no_grad():
        ta = torch.log2(torch.tensor(range(2, args.eval_max_item + 2))).to(args.device)
        ta = 1/ta
        ta_sum = torch.zeros(len(ta))
        ta_sum[0] = ta[0]
        for i in range(1, len(ta)):
            ta_sum[i] = ta_sum[i-1] + ta[i]
        ta = ta.to(args.device)
        ta_sum = ta_sum.to(args.device)
        hr = np.zeros(len(K)).astype(int)
        ndcg = np.zeros(len(K)).astype(int)
        reward_curve = np.zeros(args.eval_max_item)
        div = 0
        for i_batch, batch in enumerate(tqdm.tqdm(loader)):
            users = batch
            batch_size = len(users)
            test_env.reset(args, batch_size)
            users = users.long()
            batch_nbr = test_env.all_nbr[users].clone().to(args.device)
            batch_nbr_iid = test_env.all_nbr_iid[users].clone().to(args.device)
            batch_nbr_iid_rate = test_env.all_nbr_iid_rate[users].clone().to(args.device)
            batch_ava_item = test_env.ava_item.clone().to(args.device)
            t0 = time()
            for t_step in range(args.eval_max_item):
                action_id, action = agent.choose_action(users.to(args.device), test_env.memory[users].to(args.device),
                                                        test_env.memory_rate[users].to(args.device), batch_nbr,
                                                        batch_nbr_iid,
                                                        batch_nbr_iid_rate,
                                                        batch_ava_item,
                                                        test_env.ava_item_mask.clone().to(args.device),
                                                        None, True)

                test_env.step(users, action_id, action)
            t1 = time()
            test_env.GT_item_emb=test_env.GT_item_emb.to(args.device)

            B_ava_item_embedding = test_env.GT_item_emb[batch_ava_item].to(args.device)

            B_user_embedding = test_env.GT_user_emb[users].unsqueeze(1)

            B_user_embedding = B_user_embedding.to(args.device)

            B_ava_reward = (torch.sum(B_ava_item_embedding * B_user_embedding, dim=-1) > 0.5).long()
            B_ava_pos_num = torch.sum(B_ava_reward, dim=-1).unsqueeze(0).to(args.device)

            for i, k in enumerate(K):
                hr_nor = torch.tensor(k).expand(batch_size).to(args.device)
                ndcg_nor = ta_sum[hr_nor - 1]

                tmp_hit = test_env.res_rate[:,:k].sum(-1).to(args.device)
                tmp_dcg = (test_env.res_rate[:, :k].to(args.device)*ta[:k].unsqueeze(0)).sum(-1)


                tmp_hr = tmp_hit/hr_nor
                tmp_ndcg = tmp_dcg/ndcg_nor
                tmp_hr = tmp_hr.sum().to('cpu').numpy()
                tmp_ndcg = tmp_ndcg.sum().to('cpu').numpy()
                hr[i] += tmp_hr
                ndcg[i] += tmp_ndcg
            B_res_action = test_env.res_action
            batch_div = 0
            for i in range(1, args.eval_max_item):
                tmp_div = torch.zeros(batch_size)
                for j in range(i):
                    tmp_div += get_entropy(B_res_action[:, i], B_res_action[:, j]).to('cpu')
                tmp_div /= i
                tmp_div = tmp_div.sum().numpy()
                batch_div += tmp_div
            batch_div /= args.eval_max_item
            div += batch_div
            t3 = time()
            reward_curve += test_env.res_rate.sum(dim=0).to('cpu').numpy()
        hr = hr/len(eval_users)
        ndcg = ndcg/len(eval_users)
        div = div/len(eval_users)
        for i in range(1,args.eval_max_item):
            reward_curve[i] += reward_curve[i-1]
        reward_curve = reward_curve/len(eval_users)
        fin_div = div
        if pre == True:
            print_res = 'pre train Step {:6d}'.format(last_step)
        else:
            print_res = 'reinforce train Step {:6d}'.format(last_step)
        gg = ''
        print_res += 'div={:.4f}'.format(fin_div)
        gg += '{:.4f},'.format(fin_div)
        for i, k in enumerate(K):
            print_res += 'hr{:d}={:.4f} '.format(k, hr[i])
            gg+= '{:.4f},'.format(hr[i])
        for i, k in enumerate(K):
            print_res += 'ndcg{:d}={:.4f} '.format(k, ndcg[i])
            gg += '{:.4f},'.format(ndcg[i])

        print_res += 'LR={:6f}'.format(agent.lr_scheduler.get_lr())
        logger.info(print_res)
        logger.info(gg)
        if pre == False:
            for i, k in enumerate(K):
                writer.add_scalar(f'{ty}/hr{k}', hr[i], last_step)
                writer.flush()
                writer.add_scalar(f'{ty}/ndcg{k}', ndcg[i], last_step)
                writer.flush()
            writer.add_scalar(f'{ty}/div', fin_div, last_step)
            writer.flush()
        else:
            for i, k in enumerate(K):
                writer.add_scalar(f'pre{ty}/hr{k}', hr[i], last_step)
                writer.flush()
                writer.add_scalar(f'pre{ty}/ndcg{k}', ndcg[i], last_step)
                writer.flush()
            writer.add_scalar(f'pre{ty}/div', fin_div, last_step)
            writer.flush()
        F_measure = (2*div*hr[-2])/(div+hr[-2])
        return hr, ndcg, div, F_measure, reward_curve

def main(args):
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    device = args.device

    args.model_path = args.model_path + f'{args.model}/'
    make_dir(args.log_path, args.model_path, f'logs/{args.dataset}/')
    make_dir(f'../envs/saved_models/{args.dataset}/tmp/')

    tensorboard_path = args.log_path + curr_time + f'reward_decay{args.reward_decay}replay{args.replay}gamma{args.gamma}' \
                                                   f'target_update{args.target_update}eps{args.epsilon_start}{args.dataset}' \
                                                   f'_{args.model}_cou{args.counter}_bs{args.batch_size}_lr{args.lr}_drop{args.droprate}'

    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)

    writer = SummaryWriter(tensorboard_path)
    logger = get_logger(os.path.join('logs',
                                     f'{args.dataset}/' + f'reward_decay{args.reward_decay}replay{args.replay}gamma{args.gamma}' \
                                                          f'target_update{args.target_update}eps{args.epsilon_start}{args.dataset}' \
                                                          f'_{args.model}_cou{args.counter}_bs{args.batch_size}_lr{args.lr}_drop{args.droprate}' + f'_{curr_time}.log'))

    logger.info(args)
    user_num = args.user_num
    item_num = args.item_num
    K = [20, 50]
    data_path = f'../datasets/{args.dataset}/'
    all_data = np.load(
        data_path + f'{args.use_all_Z}Z_{args.time_interval_size_day}day_{args.valid_scale}valid_scale_{args.seq_len}seq_len_processed_data.npz',
        allow_pickle=True)
    self_nbr_item_sets = np.load(data_path + f'{args.valid_scale}self_nbr_item_set.npy', allow_pickle=True)[()]
    eval_users = all_data['eval_users']

    train_uid, user_train, all_nbr, Z_d, all_valid_per_num, user_valid, train_mat, all_nbr_iid, all_nbr_iid_rate, ts = \
        all_data['train_uid'], \
        all_data['user_train'][()], \
        all_data['train_nbr'], \
        all_data['train_Z_d'], \
        all_data['all_valid_per_num'][()], \
        all_data['all_valid'][()], \
        all_data['train_matrix'][()], \
        all_data['train_nbr_iid'], \
        all_data['train_nbr_iid_rate'], \
        all_data['train_ts']

    norm_i_embedding = torch.from_numpy(np.load(data_path + 'ground_truth_item_emb.npy'))

    Z_bin_data = np.load(
        data_path + f'{args.time_interval_size_day}day_{args.valid_scale}valid_scale_Z_bin.npz',
        allow_pickle=True)
    Z_bin = Z_bin_data['train_Z_bin']
    Z_time_interval_size = Z_bin_data['time_interval_size']
    Z_min_time = Z_bin_data['min_time']
    Z_max_time = Z_bin_data['max_time']

    print("load fin")

    train_loader = DataLoader(dataset=TrainDataset(all_data, item_num),
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=False)

    eval_loader = DataLoader(dataset=newEvalDataset(all_data, item_num),
                             batch_size=args.eval_batch_size,
                             num_workers=args.num_workers,
                             shuffle=False,
                             drop_last=False)
    from envs.counter_env import Env, TestEnv



    for r in range(args.repeat):
        seed = args.seed + r
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        agent = DMIR(args)

        last_step = 0
        CA_path = args.CA_path
        env = Env(train_mat, user_train,
                  all_nbr, all_nbr_iid, all_nbr_iid_rate, self_nbr_item_sets,
                  Z_d, Z_bin, Z_time_interval_size, Z_min_time, Z_max_time,
                  ts, args, CA_path)

        CA_opt = torch.optim.Adam(env.counterfactor.parameters(), lr=args.CA_lr)


        torch.cuda.empty_cache()

        CA_path = f'../envs/saved_models/{args.dataset}/tmp/{args.model}{curr_time}updated_CA.pth'
        best_F = 0
        best_div = 0
        best_hr = np.zeros(len(K))
        best_ndcg = np.zeros(len(K))
        for ep in range(1, args.epoch + 1):
            env, step = train(args, train_loader, eval_loader, env, agent, writer, eval_users, last_step,
                              user_valid,
                              logger, norm_i_embedding, curr_time, CA_opt)
            torch.cuda.empty_cache()
            test_env = TestEnv(user_num, env.memory, env.memory_rate, env.all_nbr,
                               env.all_nbr_iid,
                               env.all_nbr_iid_rate, args)
            hr, ndcg, div, F_measure, reward_curve = eval(args, eval_loader, test_env, agent, writer, ep,
                                                          eval_users, norm_i_embedding, K, logger)
            if F_measure > best_F:
                best_hr = hr
                best_ndcg = ndcg
                best_div = div
                best_F = F_measure
                agent.save(args, args.model_path)
                np.save(f'../datasets/{args.dataset}/{args.model}memory.npy',env.memory.numpy())
                np.save(f'../datasets/{args.dataset}/{args.model}memory_rate.npy', env.memory_rate.numpy())
                np.save(f'../datasets/{args.dataset}/{args.model}reward_curve.npy', reward_curve)
            torch.cuda.empty_cache()

            last_step = step
            logger.info(f'epoch {ep} fin')
            env.reset(train_mat, user_train,
                      all_nbr, all_nbr_iid, all_nbr_iid_rate, self_nbr_item_sets,
                      Z_d, Z_bin, Z_time_interval_size, Z_min_time, Z_max_time,
                      ts, args, CA_path)
        print("F-measure:{:.4f} Diversity:{:.4f} HR@20:{:.4f} HR@50:{:.4f} NDCG@20:{:.4f} NDCG@50:{:.4f}".
              format(best_F, best_div, best_hr[0], best_hr[1], best_ndcg[0], best_ndcg[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TEA-Gsage')
    parser.add_argument('--model', type=str, default='CAR')
    parser.add_argument('--dataset', type=str, default='Ciao')
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--memory_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_gamma', type=float, default=0.05)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--l2rg', type=float, default=1e-6)
    parser.add_argument('--droprate', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=500)
    parser.add_argument('--soft_tau', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--check_step', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_item', type=int, default=20)
    parser.add_argument('--eval_max_item', type=int, default=50)
    parser.add_argument('--buffer_size', type=int, default=50000)
    parser.add_argument('--update_size', type=int, default=10000)
    parser.add_argument('--emb_reg', type=float, default=5e-4)
    parser.add_argument('--epsilon_start', type=float, default=0.3)
    parser.add_argument('--epsilon_end', type=float, default=0)
    parser.add_argument('--epsilon_decay', type=float, default=5)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--user_num', type=int, default=2342 + 1)
    parser.add_argument('--item_num', type=int, default=77540 + 1)
    parser.add_argument('--target_update', type=int, default=1000)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--use_pos_emb', type=bool, default=True)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--Z_top', type=float, default=0.5)
    parser.add_argument('--fake', type=int, default=0)
    parser.add_argument('--valid_scale', type=float, default=0.5)
    parser.add_argument('--reward_decay', type=float, default=0.9)
    parser.add_argument('--train_reward_decay', type=float, default=0.9)

    parser.add_argument('--log_path', type=str, default='../RES/Ciao/')
    parser.add_argument('--model_path', type=str, default='../model_dict/Ciao/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_all_Z', type=int, default=1)
    parser.add_argument('--time_interval_size_day', type=int, default=30)
    parser.add_argument('--pos_line', type=int, default=4)

    parser.add_argument('--CA_lr', type=float, default=0.0001)
    parser.add_argument('--CA_edim', type=int, default=64)
    parser.add_argument('--CA_drop', type=float, default=0.3)
    parser.add_argument('--CA_path', type=str, default='')
    parser.add_argument('--rate_edim_pa', type=int, default=1)
    parser.add_argument('--Z_edim_pa', type=int, default=1)
    parser.add_argument('--beta_edim_pa', type=int, default=8)
    parser.add_argument('--counter', type=int, default=0)

    parser.add_argument('--replay', type=int, default=1)
    parser.add_argument('--pre_train', type=int, default=0)
    parser.add_argument('--pre_batch_size', type=int, default=1024)
    parser.add_argument('--pre_epoch', type=int, default=5000)
    parser.add_argument('--pre_check_epoch', type=int, default=20)
    parser.add_argument('--pre_start_epoch', type=int, default=0)
    parser.add_argument('--pre_lr', type=float, default=0.001)
    parser.add_argument('--pre_patience', type=int, default=5)
    parser.add_argument('--check_line', type=float, default=0.9)
    parser.add_argument('--use_att', type=int, default=0)
    parser.add_argument('--nbr_wei',type=float,default=1)
    parser.add_argument('--neg_wei',type=float,default=1)
    parser.add_argument('--use_nbr',type=int,default=1)


    args = parser.parse_args()
    if args.dataset == 'Epin':
        args.user_num = 18088 + 1
        args.item_num = 261649 + 1
        args.check_line = 0.95
        args.epsilon_start = 0.5
        args.nbr_wei = 0.0000001
    elif args.dataset == 'Yelp':
        args.user_num = 332132 + 1
        args.item_num = 197230 + 1
        args.check_line = 0.9
        args.epsilon_start = 0.7
        args.nbr_wei = 0.0000001

    if args.dataset != 'Ciao':
        args.log_path = f'../RES/{args.dataset}/'
        args.model_path = f'../model_dict/{args.dataset}/'
    main(args)

