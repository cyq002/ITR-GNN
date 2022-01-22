#!/usr/bin/python3
# -*-coding:UTF-8-*-
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
import random

import model


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        if isinstance(variable, model.Model):
            variable = nn.DataParallel(variable)    # gpu_ids
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def handle_data(all_sess, train_len=None):
    sess_len = [len(now_sess) for now_sess in all_sess]
    if train_len is None:
        max_len = max(sess_len)
    else:
        max_len = train_len
        sess_len = [min(l, max_len) for l in sess_len]
    print('max_len:' + str(max_len))
    sess_padded = [list(sess) + [0] * (max_len - length) if length < max_len else list(sess[-max_len:]) for sess, length in zip(all_sess, sess_len)]
    sess_mask = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len for le in sess_len]
    return sess_padded, sess_mask, max_len, sess_len


class Data(Dataset):
    def __init__(self, opt, data, soft_label=None, train_len=None, train=True):
        self.opt = opt
        self.train = train
        inputs, mask, max_len, len_data = handle_data(data[0], train_len)
        self.len_data = len_data
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        if self.train and self.opt.train_student:
            self.soft_score, self.soft_label = soft_label[0], soft_label[1]
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len

    def __getitem__(self, index):
        u_input, mask, target, s_len = self.inputs[index], self.mask[index], self.targets[index], self.len_data[index]
        if self.train and self.opt.train_student:
            soft_target = np.zeros(self.opt.num_node - 1)  # [43097]
            soft_score, soft_label = self.soft_score[index], self.soft_label[index]
            for score, label in zip(soft_score, soft_label):
                soft_target[label - 1] = score
        max_n_node = self.max_len
        pos = torch.arange(0, max_n_node).long()
        pos[0:s_len] = reversed(pos[0:s_len])
        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
        adj = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            adj[v][v] = 1
            adj[u][v] = 1

        graph_mask = np.zeros((max_n_node, max_n_node))
        graph_mask[:s_len, :s_len] = 1
        if self.train and self.opt.train_student:
            return [torch.tensor(mask), torch.tensor(target), torch.tensor(adj), torch.tensor(items),
                    torch.tensor(alias_inputs), pos, torch.tensor(graph_mask), torch.tensor(u_input), torch.tensor(soft_target)]
        else:
            return [torch.tensor(mask), torch.tensor(target), torch.tensor(adj), torch.tensor(items), torch.tensor(alias_inputs), pos, torch.tensor(graph_mask), torch.tensor(u_input)]

    def __len__(self):
        return self.length
