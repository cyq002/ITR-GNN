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
parser.add_argument('--dataset', default='Tmall', help='diginetica/Tmall')
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
parser.add_argument('--save_path', help='log file path')
parser.add_argument('--num_node', type=int)
parser.add_argument('--smooth', type=float, default=0.2)
# parser.add_argument('--h', type=int, default=4)
parser.add_argument('--train_teacher', action='store_true', help='train teacher model')
parser.add_argument('--train_student', action='store_true', help='train student model')
parser.add_argument('--pretrained_teacher_path', type=str, default='teacher_top10_Tmall_epoch_10.pth')
parser.add_argument('--top', type=int, default=5, help='rank top as high confidence sample')
parser.add_argument('--lambda_', type=float, default=0.1, help='soft label loss and hard label loss')
opt = parser.parse_args()


def forward(model, data, train=True):
    if opt.train_student and train:
        mask, target, adj, items, alias_inputs, pos, graph_mask, _, soft_target = data
    else:
        mask, target, adj, items, alias_inputs, pos, graph_mask, _ = data

    mask = trans_to_cuda(mask).long()
    graph_mask = trans_to_cuda(graph_mask).long()
    adj = trans_to_cuda(adj).long()
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    pos = trans_to_cuda(pos).long()
    hidden = model(mask, adj, items, alias_inputs, pos, graph_mask)
    get_score = model.module.get_score if hasattr(model, 'module') else model.get_score
    if opt.train_student and train:
        return target, get_score(hidden), soft_target
    else:
        return target, get_score(hidden)


def test(model, test_data):
    model.eval()
    batch_size = model.module.batch_size if hasattr(model, 'module') else model.batch_size
    test_loader = DataLoader(test_data, num_workers=opt.num_workers, batch_size=batch_size, shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    for data in test_loader:
        targets, scores = forward(model, data, False)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target in zip(sub_scores, targets):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)
    return result


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    batch_size = model.module.batch_size if hasattr(model, 'module') else model.batch_size
    train_loader = DataLoader(train_data, num_workers=opt.num_workers, batch_size=batch_size, shuffle=True, pin_memory=True)
    # batch
    for data in tqdm(train_loader):
        model.module.optimizer.zero_grad() if hasattr(model, 'module') else model.optimizer.zero_grad()
        if opt.train_student:
            targets, scores, soft_targets = forward(model, data, True)
        else:
            targets, scores = forward(model, data, True)
        targets = trans_to_cuda(targets).long()
        loss = model.module.loss_function(scores, targets - 1)
        # loss = model.loss_function(scores, targets - 1)
        if opt.train_student:
            soft_targets = trans_to_cuda(soft_targets).float()
            soft_loss = model.module.sl_loss_function(scores, soft_targets)
            loss = (1 - opt.lambda_) * loss + opt.lambda_ * soft_loss
        loss.backward()
        model.module.optimizer.step()
        # model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)
    logging.info('total loss:' + str(total_loss))
    model.module.scheduler.step()
    # model.scheduler.step()
    logging.info('start predicting...')
    print('start predicting: ', datetime.datetime.now())
    result = test(model, test_data)

    return result


def main():
    opt.save_path = join('log', opt.dataset,
                         str(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')))
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    logging.basicConfig(format='%(filename)s [%(asctime)s]  %(message)s', filename=join(opt.save_path, 'log.txt'), filemode='a', level=logging.DEBUG)
    init_seed(2020)
    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.pos_len = 69
        opt.dropout = 0.2
        opt.tr_layer = 1
        # opt.smooth = 0.0
        opt.lambda_ = 0
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.pos_len = 39
        opt.tr_layer = 2
        opt.dropout = 0.2
        # opt.smooth = 0.2
        opt.lambda_ = 0.6
    elif opt.dataset == 'last_fm':
        num_node = 38616
        opt.pos_len = 19
        opt.tr_layer = 3
        opt.dropout = 0.2
        # opt.smooth = 0.1
        opt.lambda_ = 0.5
    else:
        return
    opt.num_node = num_node
    logging.info(opt)
    if opt.train_teacher:
        train_data = pickle.load(open('dataset/' + opt.dataset + '/high_confidence_train_top' + str(opt.top) + '.txt', 'rb'))
    else:
        train_data = pickle.load(open('dataset/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        logging.info('split validation set...')
        print("split validation set")
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
        logging.info('over...')
    else:
        test_data = pickle.load(open('dataset/' + opt.dataset + '/test.txt', 'rb'))

    soft_label = None
    if opt.train_student:
        soft_label = pickle.load(open('dataset/' + opt.dataset + '/soft_label_top' + str(opt.top) + '.txt', 'rb'))

    train_data = Data(opt, train_data, soft_label, train_len=opt.train_len, train=True)
    test_data = Data(opt, test_data, None, train_len=opt.train_len, train=False)
    model = trans_to_cuda(Model(num_node, opt))
    if opt.train_teacher:
        module_list = torch.load(opt.pretrained_teacher_path)
        model.module.load_state_dict(module_list)
        print('Loaded Model..')
    print(opt)
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    logging.info('training...')
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        logging.info('epoch:' + str(epoch))
        epoch_start = time.time()
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        checkpoint = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(checkpoint, join(opt.save_path, ('teacher_top' + str(opt.top) + '_' if opt.train_teacher else '') + opt.dataset + '_epoch_' + str(epoch + 1) + '.pth'))
        logging.info('Current Result:\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit, mrr))
        print('Current Result\tEpoch:\t%d:' % (epoch, ))
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit, mrr))
        logging.info('Best Result:\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
        epoch_end = time.time()
        logging.info("Run time: %f s" % (epoch_end - epoch_start))
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
# python main.py --dataset=Tmall --epoch=6 --patience=6 --smooth=0
# python generate_high_confidence_data.py --dataset=Tmall --top=5 --pretrained_teacher_path=''
# python main.py --dataset=Tmall --epoch=6 --patience=6 --train_teacher --top=5 --smooth=0 --pretrained_teacher_path=''
# python generate_soft_label.py --dataset=Tmall --teacher_path='' --top=5
# python main.py --dataset=Tmall --train_student --top=5 --smooth=0.2
