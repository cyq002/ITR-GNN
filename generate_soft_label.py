#!/usr/bin/python3
# -*-coding:UTF-8-*-
import time
import argparse
import pickle
from model import *
from utils import *
from torch.utils.data import DataLoader
import logging
import torch
import os
from tqdm import tqdm
import datetime
import numpy as np
import random
from os.path import join
# CUDA_VISIBLE_DEVICES=1 python my_script.py


# set random seed
def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Tmall')
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--tr_layer', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--train_len', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--pos_len', type=int, default=69)
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--smooth', type=float, default=0.1)
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--teacher_path', type=str, default='teacher_top10_Tmall_epoch_6.pth')
opt = parser.parse_args()


def forward(model, data):
    mask, target, adj, items, alias_inputs, pos, graph_mask, _ = data
    mask = trans_to_cuda(mask).long()
    graph_mask = trans_to_cuda(graph_mask).long()
    adj = trans_to_cuda(adj).long()
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    pos = trans_to_cuda(pos).long()
    hidden = model(mask, adj, items, alias_inputs, pos, graph_mask)
    return target, model.module.get_score(hidden)


def test(model, test_data):
    model.eval()
    test_loader = DataLoader(test_data, num_workers=opt.num_workers, batch_size=model.module.batch_size, shuffle=False, pin_memory=True)
    score, soft_label = [], []
    for data in tqdm(test_loader):
        _, scores = forward(model, data)
        sub_scores, label = scores.topk(opt.top)
        label = label + 1
        label = trans_to_cpu(label).detach().numpy()
        sub_scores = torch.softmax(sub_scores, -1)
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        soft_label += list(label)
        score += list(sub_scores)
    pickle.dump((score, soft_label), open('dataset/' + opt.dataset + '/soft_label_top' + str(opt.top) + '.txt', 'wb'))


def main():
    init_seed(2020)
    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.pos_len = 69
        opt.dropout = 0.2
        opt.tr_layer = 1
        opt.smooth = 0.0
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.pos_len = 39
        opt.tr_layer = 2
        opt.dropout = 0.2
        opt.smooth = 0.2
    elif opt.dataset == 'last_fm':
        num_node = 38616
        opt.pos_len = 19
        opt.tr_layer = 3
        opt.dropout = 0.2
        opt.smooth = 0.1
    else:
        return
    train_data = pickle.load(open('dataset/' + opt.dataset + '/train.txt', 'rb'))
    train_data = Data(opt, train_data, train_len=opt.train_len, train=False)
    model = trans_to_cuda(Model(num_node, opt))
    module_list = torch.load(opt.teacher_path)
    model.module.load_state_dict(module_list)
    print('Loaded Teacher Model..')
    print(opt)
    test(model, train_data)


if __name__ == '__main__':
    main()

