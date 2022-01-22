#!/usr/bin/python3
# -*-coding:UTF-8-*-
from torch import nn
import torch
import math
import torch.nn.functional as F


class SLCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SLCrossEntropyLoss, self).__init__()

    def forward(self, p, target):
        log_p = F.log_softmax(p, dim=1)  # softmax + log
        loss = -1 * torch.sum(target * log_p, 1)
        return loss.mean()


class LSCrossEntropyLoss(nn.Module):
    def __init__(self, label_smooth=None, cls=None):
        super(LSCrossEntropyLoss, self).__init__()
        self.label_smooth = label_smooth
        self.cls = cls

    def forward(self, p, target):
        if self.label_smooth is not None:
            log_p = F.log_softmax(p, dim=1)  # softmax + log
            target = F.one_hot(target, self.cls)  # one-hot
            target = torch.clamp(target.float(), min=self.label_smooth / (self.cls - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * log_p, 1)
        else:
            loss = F.cross_entropy(p, target, reduction='mean', ignore_index=-100)
        return loss.mean()


class TRLayer(nn.Module):
    def __init__(self, opt):
        super(TRLayer, self).__init__()
        self.opt = opt
        self.tr_layer = opt.tr_layer
        self.hidden_size = opt.hidden_size
        self.linear1 = nn.Linear(2 * self.hidden_size, 1)

    # 有向无权图
    def aggregate(self, hidden, adj_out):
        batch_size = hidden.shape[0]
        s_len = hidden.shape[1]

        attention = self.linear1(torch.cat(
            [hidden.repeat(1, 1, s_len).view(batch_size, s_len * s_len, self.hidden_size), hidden.repeat(1, s_len, 1)],
            -1)).squeeze(-1).view(batch_size, s_len, s_len)
        # out
        attention_out = torch.tanh(attention)
        # in
        attention_in = torch.transpose(torch.clone(attention_out), -1, -2)
        adj_in = torch.transpose(torch.clone(adj_out), -1, -2)
        item_mask = torch.zeros_like(attention_in)
        alpha_out = torch.where(adj_out.eq(1), attention_out, item_mask)
        alpha_in = torch.transpose(torch.clone(alpha_out), -1, -2)
        alpha = (alpha_out + alpha_in) / 2
        adj = adj_out + adj_in
        item_mask = -9e15 * torch.ones_like(attention_out)
        alpha = torch.where(adj.eq(0), item_mask, alpha)
        alpha = torch.softmax(alpha, dim=-1)
        hidden = torch.matmul(alpha, hidden)
        return hidden

    def forward(self, hidden, adj_out, alias_inputs):
        for i in range(self.tr_layer):
            hidden = self.aggregate(hidden, adj_out)
        get_h = lambda index: hidden[index][alias_inputs[index]]
        hidden = torch.stack([get_h(i) for i in torch.arange(len(alias_inputs)).long()])
        return hidden


class Model(nn.Module):
    def __init__(self, num_node, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.pos_len = opt.pos_len
        self.num_node = num_node
        self.batch_size = opt.batch_size
        self.dropout = opt.dropout
        self.hidden_size = opt.hidden_size
        self.lr = opt.lr
        self.step_size = opt.lr_step
        self.gamma = opt.lr_dc
        self.alpha = opt.alpha
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.kernel_size = opt.kernel_size
        self.label_smooth = opt.smooth

        self.tr_layer = TRLayer(opt)
        self.embedding = nn.Embedding(self.num_node, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.pos_len, self.hidden_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear4 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.b_2 = nn.Parameter(torch.Tensor(self.hidden_size))

        self.conv_q = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=self.hidden_size)
        self.conv_k = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=self.hidden_size)
        self.conv_v = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=self.hidden_size)

        self.loss_function = LSCrossEntropyLoss(label_smooth=self.label_smooth, cls=self.num_node-1)
        self.sl_loss_function = SLCrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
        self.init_parameters()

    def init_parameters(self):
        stand = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stand, stand)

    def get_score(self, intent):
        b = self.embedding.weight[1:]
        scores = torch.matmul(intent, b.transpose(1, 0))
        return scores

    def forward(self, mask, adj, items, alias_inputs, pos, graph_mask):
        items_h = self.embedding(items)
        hidden = self.tr_layer(items_h, adj, alias_inputs)

        s_len = hidden.shape[1]
        pos_emb = self.position_embedding(pos)
        pos_emb = pos_emb.permute(0, 2, 1)
        p_q, p_k, p_v = self.conv_q(pos_emb), self.conv_k(pos_emb), self.conv_v(pos_emb)
        p_q, p_k, p_v = p_q.permute(0, 2, 1), p_k.permute(0, 2, 1), p_v.permute(0, 2, 1)
        p_q, p_k, p_v = self.leaky_relu(p_q), self.leaky_relu(p_k), self.leaky_relu(
            p_v)
        pos_mask = graph_mask
        dk = pos_emb.size()[-1]
        p_alpha = p_q.matmul(p_k.transpose(-2, -1)) / math.sqrt(dk)
        p_alpha = p_alpha.masked_fill(pos_mask == 0, -9e15)
        p_alpha = F.softmax(p_alpha, dim=-1)
        pos_emb = p_alpha.matmul(p_v)

        nh = hidden + F.dropout(pos_emb, self.dropout)
        h_n = nh[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        h_n = h_n.unsqueeze(1).repeat(1, s_len, 1)
        mask = mask.float().unsqueeze(-1)
        hs = torch.sum(nh * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, s_len, 1)
        nh = self.leaky_relu(self.linear1(nh) + self.linear2(hs) + self.linear3(h_n) + self.linear4(torch.where(hs > h_n, hs, h_n)))
        nh = F.dropout(nh, self.dropout)
        beta = torch.matmul(nh, self.w_2) + self.b_2
        beta = beta * mask
        intent = torch.sum(beta * hidden, 1)
        return intent
