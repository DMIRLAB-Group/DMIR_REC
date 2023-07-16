import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from time import time
from datetime import datetime
import logging
import sys
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path

def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

class MyDataset(Dataset):
    def __init__(self, data, pos_line):
        self.user = data[:,0]
        self.item = data[:, 1]
        self.rate = data[:, -1]
        self.rate[self.rate<=pos_line] = 0
        self.rate[self.rate>pos_line] = 1

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        user = self.user[idx]
        item = self.item[idx]
        rate = self.rate[idx]

        return user, item, rate

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

class GroundTruth(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(GroundTruth, self).__init__()
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0).to(args.device)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0).to(args.device)
        nn.init.uniform_(self.user_embs.weight[1:], a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight[1:], a=-0.5 / item_num, b=0.5 / item_num)

    def forward(self, user, item):
        user_embedding = self.user_embs(user)
        item_embedding = self.item_embs(item)
        score = torch.sum((user_embedding*item_embedding), -1)

        return score

def train(model, loader, opt, args):
    model.train()
    total_loss = 0.0
    for batch in loader:
        user, item, label = batch
        user = user.to(args.device).long()
        item = item.to(args.device).long()
        label = label.float().to(args.device)
        score = model(user, item)

        loss = F.binary_cross_entropy_with_logits(score, label)
        total_loss += loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return total_loss/len(loader)

def evaluate(model, loader, args):
    model.eval()
    TP = TN = FN = FP = 0


    with torch.no_grad():
        for batch in loader:
            user, item, label = batch
            user = user.to(args.device).long()
            item = item.to(args.device).long()
            score = model(user, item).to('cpu').numpy()
            label = label.numpy()
            score[score>=0.5] = 1
            score[score<0.5] = 0
            TP += np.sum((score==label) * (label==1))
            TN += np.sum((score==label) * (label==0))
            FN += np.sum((score!=label) * (label==1))
            FP += np.sum((score != label) * (label == 0))

        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)

        return F1, acc

def main(args):
    device = args.device
    user_num = 2342 + 1
    item_num = 77540 + 1
    if args.dataset == 'Epin':
        user_num = 18088 + 1
        item_num = 261649 + 1
    if args.dataset == 'Yelp':
        user_num = 332132 + 1
        item_num = 197230 + 1
    u2ui = np.load(args.data_path + 'noiso_reid_u2ui.npz')
    u2i = u2ui['u2i']
    print('user', u2i[:, 0].min(), u2i[:, 0].max())
    df = pd.DataFrame(data=u2ui['u2i'], columns=['user', 'item', 'ts', 'rate'])
    print('Raw u2i =', df.shape)
    df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
    u2i = df.to_numpy()
    print(type(u2i),u2i.shape)
    print('user',u2i[:,0].min(),u2i[:,0].max())
    print('item', u2i[:, 1].min(), u2i[:, 1].max())
    print('ts', u2i[:, 2].min(), u2i[:, 2].max())
    print('rate', u2i[:, 3].min(), u2i[:, 3].max())

    print('Loading...')
    st = time()
    timestr = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    model_path = f'saved_models/{args.dataset}/{timestr}.pth'

    make_dir(f'saved_models/{args.dataset}')
    device = torch.device(args.device)

    metrics_list = []
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = GroundTruth(user_num, item_num, args)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        dataset=MyDataset(u2i, args.pos_line),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False)
    best_score = patience_cnt = 0
    all_item_tensor = torch.tensor(range(item_num)).to(args.device)
    for epoch in range(1, args.max_epochs+1):
        st = time()
        train_loss = train(model, loader, opt, args)
        print('Epoch:{} Train Loss={:.4f} Time={:.2f}s'.format(
            epoch, train_loss, time() - st))

        if epoch % args.check_epoch == 0 and epoch >= args.start_epoch:
            val_metrics = evaluate(model, loader, args)
            F1, acc = val_metrics

            item_embedding = model.item_embs(all_item_tensor)
            norm_i = torch.softmax(item_embedding, dim=-1)
            ze = torch.sum(norm_i == 0)

            if best_score < F1 and ze == 0:
                torch.save(model.state_dict(), model_path)
                print('Validation F1 increased: {:.4f} --> {:.4f}'.format(best_score, F1))
                best_score = F1
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt == args.patience:
                print('Early Stop!!!')
                break

    print('Testing')
    model.load_state_dict(torch.load(model_path)) # 加载模型
    all_item = torch.tensor(range(item_num)).to(args.device)
    all_user = torch.tensor(range(user_num)).to(args.device)
    with torch.no_grad():
        all_item_emb = model.item_embs(all_item)
        all_user_emb = model.user_embs(all_user)
    norm_all_item_emb = torch.softmax(all_item_emb, dim=-1).to('cpu').numpy()
    print('norm item emb shape', norm_all_item_emb.shape)
    no_ze = np.sum(norm_all_item_emb == 0)
    print('no zero', no_ze)
    all_item_emb = all_item_emb.to('cpu').numpy()
    all_user_emb = all_user_emb.to('cpu').numpy()
    np.save(f'../datasets/{args.dataset}/ground_truth_user_emb.npy', all_user_emb)
    np.save(f'../datasets/{args.dataset}/ground_truth_item_emb.npy', all_item_emb)
    np.save(f'../datasets/{args.dataset}/norm_ground_truth_item_emb.npy', norm_all_item_emb)
    test_metrics = evaluate(model, loader, args)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='Ciao')
    parser.add_argument('--data_path', type=str, default='../datasets/Ciao/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--edim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--check_epoch', type=int, default=5)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--pos_line', type=int, default=4)
    args = parser.parse_args()
    args.data_path = f'../datasets/{args.dataset}/'
    main(args)

