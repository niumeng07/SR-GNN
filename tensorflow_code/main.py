#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file main.py
# @brief ggnn
# @author myliuda@sina.com
# @date 2019-07-10


from __future__ import division
import numpy as np
from model import *
from utils import build_graph, Data, split_validation
import pickle
import argparse
import datetime
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--method', type=str, default='ggnn', help='ggnn/gat/gcn')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--model_dir', type=str, default='./model/', help='model save dir.')
parser.add_argument('--model_name', type=str, default='ggnn.ckpt', help='model name.')
opt = parser.parse_args()
#train_data, test_data, type=tuple
#数据格式: (input_seqs, real_tar.get), 即用户在一定的click后,下一次click的Item
train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
# all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
if opt.dataset == 'diginetica':
    n_node = 43098
elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
    n_node = 37484
else:
    n_node = 310 #提前计算好的
# g = build_graph(all_train_seq)
# Data类          真实数据
# train_data: list[inputs], list[targets]
train_data = Data(train_data, sub_graph=True, method=opt.method, shuffle=True)
test_data = Data(test_data, sub_graph=True, method=opt.method, shuffle=False)
model = GGNN(hidden_size=opt.hiddenSize, out_size=opt.hiddenSize, batch_size=opt.batchSize, n_node=n_node,
                 lr=opt.lr, l2=opt.l2,  step=opt.step, decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize, lr_dc=opt.lr_dc,
                 nonhybrid=opt.nonhybrid, model_dir=opt.model_dir, model_name=opt.model_name)
print(opt)
best_result = [0, 0]
best_epoch = [0, 0]
test_scores_ = []
test_item_ = []
for epoch in range(opt.epoch):
    print('epoch: ', epoch, '===========================================')
    slices = train_data.generate_batch(model.batch_size) #list[ndarray]
    fetches = [model.opt, model.loss_train, model.global_step]
    print('start training: ', datetime.datetime.now())
    loss_ = []
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
        #print(adj_in, adj_out, alias, item, mask, targets)
        _, loss, _ = model.run(fetches, targets, item, adj_in, adj_out, alias,  mask)
        loss_.append(loss)
    loss = np.mean(loss_)
    slices = test_data.generate_batch(model.batch_size)
    model.save()
    print('start predicting: ', datetime.datetime.now())
    hit, mrr, test_loss_, test_scores_ = [], [], [], []
    test_item_ = []
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets = test_data.get_slice(i)
        test_item_.append(item)
        scores, test_loss = model.run([model.score_test, model.loss_test], targets, item, adj_in, adj_out, alias,  mask)
        test_loss_.append(test_loss)
        test_scores_.append(scores)
        index = np.argsort(scores, 1)[:, -20:] #取概率最高的20个,只要这20个命中了target就算命中
        for score, target in zip(index, targets):
            ## np.isin(a, b) a的元素是否包含于b. a和b可退化为标量
            # sample: np.isin([1,2], [1,3]) returns array([True, False])
            # print(target, score) #real_target, 预估概率最高的20项
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (20-np.where(score == target - 1)[0][0]))
    hit = np.mean(hit)*100
    mrr = np.mean(mrr)*100
    test_loss = np.mean(test_loss_)
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = epoch
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1]=epoch
    #Recall和Mrr都为*100之后的
    print('train_loss: %.4f\ttest_loss: %4f\tRecall@20: %.4f\tMMR@20: %.4f\tEpoch: %d\t%d.'%
          (loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
