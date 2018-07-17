#!/usr/bin/env python3
# 演示测试效果
# 输入: query 一张图片
# 输出: gallery 中选出的图片
# @wieSeele :3

import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
import pickle
import argparse
from PIL import Image
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
import shutil

from aligned_reid.dataset import create_dataset
from aligned_reid.model.Model import Model
from aligned_reid.utils.utils import str2bool
from aligned_reid.utils.utils import load_state_dict
from aligned_reid.utils.utils import load_ckpt
from aligned_reid.utils.utils import set_devices
from aligned_reid.utils.utils import measure_time
from aligned_reid.utils.re_ranking import re_ranking
from aligned_reid.utils.distance import compute_dist
from aligned_reid.utils.distance import low_memory_matrix_op
from aligned_reid.utils.distance import local_dist


def save_typical(test_set, q_name, g_names, save_path):
    """根据图像名称找到原始图像,并将索引的图像保存起来"""
    im_path = osp.join(test_set.im_dir, q_name)
    img = Image.open(im_path)
    save_path = save_path + q_name[:-4] + '/'
    if not osp.exists(save_path):
        os.makedirs(save_path)
    img.save(save_path + '0_' + q_name)
    for name in g_names:
        im_path = osp.join(test_set.im_dir, name)
        img = Image.open(im_path)
        img.save(save_path + name)


def save_min_distance(dataset, num_query, distmat, path, max_distance = 0.2):
    if osp.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    for i in range(num_query):
        path_image = dataset.im_names[i]
        name_image = osp.basename(path_image)
        path_target = osp.join(path, ('%04d_query_' % i) + name_image)
        shutil.copy(path_image, path_target)
    for i, dists in enumerate(distmat):
        for j, dist in enumerate(dists):
            if dist <= max_distance:
                path_image = dataset.im_names[num_query + j]
                name_image = osp.basename(path_image)
                path_target = osp.join(path, ('%04d_gallery_%.4f_' % (i, dist)) + name_image)
                shutil.copy(path_image, path_target)


def save_top_k(dataset, num_query, indices, path, k = 5):
    if osp.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    for i in range(num_query):
        path_image = dataset.im_names[i]
        name_image = osp.basename(path_image)
        path_target = osp.join(path, ('%04d_query_' % i) + name_image)
        shutil.copy(path_image, path_target)
    for i, ranks in enumerate(indices):
        for j in range(k):
            # print(ranks)
            # print(dataset.im_names)
            path_image = dataset.im_names[num_query + int(ranks[j])]
            name_image = osp.basename(path_image)
            path_target = osp.join(path, ('%04d_gallery_%d_' % (i, j)) + name_image)
            shutil.copy(path_image, path_target)


def exp_summary(test_set):
    # 1. extract features
    with measure_time('Extracting feature...'):
        global_feats, local_feats, im_names, marks = test_set.extract_feat(normalize_feat = True)
    # 2. build q-g-distmat, q_ids, g_ids, q_cams, g_cams, q_idx, g_idx
    q_inds = marks == 0
    g_inds = marks == 1
    mq_inds = marks == 2

    ###########################
    # A helper function just for avoiding code duplication.
    def low_memory_local_dist(x, y):
        with measure_time('Computing local distance...'):
            x_num_splits = int(len(x) / 200) + 1
            y_num_splits = int(len(y) / 200) + 1
            z = low_memory_matrix_op(
                local_dist, x, y, 0, 0, x_num_splits, y_num_splits, verbose = True)
        return z

    ###################
    # Global Distance #
    ###################
    # query-gallery distance using global distance
    global_q_g_dist = compute_dist(global_feats[q_inds], global_feats[g_inds], type = 'euclidean')
    # query-query distance using global distance
    global_q_q_dist = compute_dist(global_feats[q_inds], global_feats[q_inds], type = 'euclidean')
    # gallery-gallery distance using global distance
    global_g_g_dist = compute_dist(global_feats[g_inds], global_feats[g_inds], type = 'euclidean')

    ##################
    # Local Distance #
    ##################
    # query-gallery distance using local distance
    local_q_g_dist = low_memory_local_dist(local_feats[q_inds], local_feats[g_inds])
    # query-query distance using local distance
    local_q_q_dist = low_memory_local_dist(local_feats[q_inds], local_feats[q_inds])
    # gallery-gallery distance using local distance
    local_g_g_dist = low_memory_local_dist(local_feats[g_inds], local_feats[g_inds])

    #########################
    # Global+Local Distance #
    #########################
    global_local_q_g_dist = global_q_g_dist + local_q_g_dist
    global_local_q_q_dist = global_q_q_dist + local_q_q_dist
    global_local_g_g_dist = global_g_g_dist + local_g_g_dist

    #########################
    # Re_Ranking #
    #########################
    # re-ranked query-gallery distance
    re_r_global_q_g_dist = re_ranking(global_q_g_dist, global_q_q_dist, global_g_g_dist)
    re_r_global_local_q_g_dist = re_ranking(re_r_global_q_g_dist, local_q_q_dist, local_g_g_dist)

    # re_r_global_local_q_g_dist = re_ranking(global_local_q_g_dist, global_local_q_q_dist, global_local_g_g_dist)

    ###########################

    # distmat = compute_dist(
    #     global_feats[q_inds], global_feats[g_inds], type = 'euclidean')
    distmat = re_r_global_local_q_g_dist
    numaxis = np.arange(len(q_inds))
    q_idx = numaxis[q_inds]
    g_idx = numaxis[g_inds]

    indices = np.argsort(distmat, axis = 1)
    num_query = len(q_idx)
    # print(distmat, indices)
    save_top_k(test_set, num_query, indices, path = 'result_5', k = 5)
    save_min_distance(test_set, num_query, distmat, path = 'result_0.2', max_distance = 0.3)
    # print(numaxis,q_idx,g_idx)


class ExtractFeature(object):
    """A function to be called in the val/test set, to extract features.
    Args:
      TVT: A callable to transfer images to specific device.
    """

    def __init__(self, model, TVT):
        self.model = model
        self.TVT = TVT

    def __call__(self, ims):
        old_train_eval_model = self.model.training
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model.eval()
        ims = Variable(self.TVT(torch.from_numpy(ims).float()))
        global_feat, local_feat = self.model(ims)[:2]
        global_feat = global_feat.data.cpu().numpy()
        local_feat = local_feat.data.cpu().numpy()
        # Restore the model to its old train/eval mode.
        self.model.train(old_train_eval_model)
        return global_feat, local_feat


def main():
    # name_data = '20180525184611494'
    name_data = '20180717103311494'

    TVT, TMO = set_devices((0,))

    dataset_kwargs = dict(
        name = name_data,
        resize_h_w = (256, 128),
        scale = True,
        im_mean = [0.486, 0.459, 0.408],
        im_std = [0.229, 0.224, 0.225],
        batch_dims = 'NCHW',
        num_prefetch_threads = 1)
    test_set_kwargs = dict(
        part = 'val',
        batch_size = 32,
        final_batch = True,
        shuffle = False,
        mirror_type = ['random', 'always', None][2],
        prng = np.random)
    test_set_kwargs.update(dataset_kwargs)
    test_set = create_dataset(**test_set_kwargs)

    with measure_time('Load model'):
        model = Model(local_conv_out_channels = 128,
                      num_classes = 1000)
        model_w = DataParallel(model)
        optimizer = optim.Adam(model.parameters(),
                               lr = 2e-4,
                               weight_decay = 0.0005)
        modules_optims = [model, optimizer]
    with measure_time('Load checkpoint'):
        load_ckpt(modules_optims, 'ckpt.pth')

    test_set.set_feat_func(ExtractFeature(model_w, TVT))

    exp_summary(test_set)


if __name__ == '__main__':
    main()
