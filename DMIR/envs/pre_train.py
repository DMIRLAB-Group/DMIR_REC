import argparse
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
from time import time
import torch.multiprocessing
from utils import parse_sampled_batch,load_ds,get_logger
from CA_model import CA
from pathlib import Path

def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def train(model, opt, train_loader, args):
    model.train()
    total_loss = 0.0
    num = 0
    for batch in train_loader:

        parsed_batch, indices, pos_num, neg_num = parse_sampled_batch(batch)
        opt.zero_grad()
        loss = model.forward(parsed_batch, indices)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        num += 1


    return total_loss / len(train_loader)


def main(args):
    user_num = 2342 + 1
    item_num = 77540 + 1
    if args.dataset == 'Epin':
        user_num = 18088 + 1
        item_num = 261649 + 1
    elif args.dataset == 'Yelp':
        user_num = 332132 + 1
        item_num = 197230 + 1

    print('Loading...')
    st = time()
    train_loader, val_loader,  user_train, eval_users = load_ds(args, item_num)
    print('Loaded {} dataset with {} users {} items in {:.2f}s'.format(args.dataset, user_num, item_num, time()-st))
    timestr = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    model_path = f'{args.model_path}{args.dataset}/{args.model}_bs{args.batch_size}_lr{args.lr}_edim{args.edim}_seq{args.seq_len}_useallZ{args.use_all_Z}_day{args.time_interval_size_day}_scale{args.valid_scale}_drop{args.droprate}_{timestr}.pth'
    make_dir(args.log_path+args.dataset+'/', args.model_path+args.dataset+'/', f'{args.log_path}{args.dataset}')
    device = torch.device(args.device)

    for r in range(args.repeat):
        seed = args.seed + r
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = CA(user_num, item_num, args.Z_top, args.edim, args.droprate, args.dataset, args.seq_len, args.device, args)
        model = model.to(device)
        opt = torch.optim.Adam(model.get_parameters(), lr=args.lr, weight_decay=args.l2rg)
        for epoch in tqdm(range(1, args.max_epochs+1)):
            st = time()
            train_loss = train(model, opt, train_loader, args)
        torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='Ciao')
    parser.add_argument('--model', default='CounterFactual')

    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--seq_num', type=int, default=10)
    parser.add_argument('--Z_top', type=float, default=0.5)
    parser.add_argument('--rate_edim_pa', type=int, default=1)
    parser.add_argument('--Z_edim_pa', type=int, default=1)
    parser.add_argument('--beta_edim_pa', type=int, default=8)

    parser.add_argument('--batch_size', type=int, default=1024, help='fixed, or change with sampled train_batches')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--l2rg', type=float, default=5e-4)
    parser.add_argument('--emb_reg', type=float, default=5e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_all_Z', type=int, default=1)
    parser.add_argument('--time_interval_size_day', type=int, default=30)
    parser.add_argument('--valid_scale', type=float, default=0.5)
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--model_path', type=str, default='./saved_models/')
    parser.add_argument('--pos_line', type=int, default=4)
    parser.add_argument('--check_line',type=float, default=0.5)
    args = parser.parse_args()
    if args.dataset == 'Epin':
        args.user_num = 18088 + 1
        args.item_num = 261649 + 1
    elif args.dataset == 'Yelp':
        args.user_num = 332132 + 1
        args.item_num = 197230 + 1
        args.lr = 0.001
        args.droprate = 0.8
    main(args)