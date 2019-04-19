# ----------------------------------------------------------
#
# Project name: DMCRID-MulCAM-reID
# File name: demo.py
#
# Last modified: 2018-12-20
#
# ----------------------------------------------------------

import argparse
import scipy.io
import torch
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')

from torchvision import datasets
from utils import render_result, sort_image

#######################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default='reid-PCB', type=str,
                    help='the saved model path')
parser.add_argument('--query-index', default=20, type=int,
                    help='index of the test image')
parser.add_argument('--data-dir', default='./datasets/Market-1501/market-reid',
                    type=str,
                    help='the testing data')
parser.add_argument('--num-images', default=5, type=int,
                    help='number of images query')
args = parser.parse_args()

image_dsets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x)) for x in
               ['gallery', 'query']}

######################################################################

fname_single = args.model_name + '_last' + '_single_query.mat'
result = scipy.io.loadmat(os.path.join('./results', fname_single))
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

fname_multi = args.model_name + '_last' + '_multi_query.mat'
multi = os.path.isfile(os.path.join('./results', fname_multi))

if multi:
    m_result = scipy.io.loadmat(os.path.join('./results', fname_multi))
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

idx = args.query_index
index = sort_image(query_feature[idx], query_label[idx], query_cam[idx],
                   gallery_feature, gallery_label, gallery_cam)

#######################################################################
#
# Visualize the rank result
#
######################################################################
query_path, _ = image_dsets['query'].imgs[idx]
query_label = query_label[idx]
print('Path to the query image:', query_path)
print('Top {} images are as follow:'.format(args.num_images))
try:
    # Visualize Ranking Result
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, args.num_images + 1, 1)
    ax.axis('off')
    render_result(query_path, 'query')
    for i in range(args.num_images):
        ax = plt.subplot(1, args.num_images + 1, i + 2)
        ax.axis('off')
        img_path, _ = image_dsets['gallery'].imgs[index[i]]
        label = gallery_label[index[i]]
        render_result(img_path)
        if label == query_label:
            ax.set_title('%d' % (i + 1), color='green')
        else:
            ax.set_title('%d' % (i + 1), color='red')
        print(img_path)
except RuntimeError:
    for i in range(args.num_images):
        img_path = image_dsets.imgs[index[i]]
        print(img_path[0])

fig.savefig("./results/demo.png")
