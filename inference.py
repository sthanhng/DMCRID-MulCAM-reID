# ----------------------------------------------------------
#
# Project name: DMCRID-MulCAM-reID
# File name: inference.py
#
# Last modified: 2018-11-21
#
# ----------------------------------------------------------


from __future__ import print_function, division

import argparse
import os
import scipy.io
import torch
import torch.nn as nn

from torchvision import datasets, transforms

from model import ResNet50, PCB, PCB_test
from utils import load_model, get_id, extract_feature

######################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default='last', type=str,
                    help='')
parser.add_argument('--data-dir', default='./datasets/Market-1501/market-reid',
                    type=str, help='the testing data')
parser.add_argument('--model-name', default='reid-ResNet50', type=str,
                    help='the saved model path')
parser.add_argument('--batch-size', default=32, type=int, help='batch-size')
parser.add_argument('--num-workers', default=8, type=int, help='num-workers')
parser.add_argument('--PCB', action='store_true', default=False,
                    help='use PCB-based model')
parser.add_argument('--multi', action='store_true', default=False,
                    help='use multiple query')
args = parser.parse_args()


#####################################################################
#
# Load Data
#
#####################################################################
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Data transforms
data_transforms = transforms.Compose([
    transforms.Resize((288, 144), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

if args.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

# Multiple query
if args.multi:
    image_dsets = {
        x: datasets.ImageFolder(os.path.join(args.data_dir, x),
                                data_transforms)
        for x in ['gallery', 'query', 'multi-query']
    }
    dset_loaders = {
        x: torch.utils.data.DataLoader(image_dsets[x],
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=args.num_workers)
        for x in ['gallery', 'query', 'multi-query']}
else:
    image_dsets = {
        x: datasets.ImageFolder(os.path.join(args.data_dir, x),
                                data_transforms)
        for x in ['gallery', 'query']}
    dset_loaders = {
        x: torch.utils.data.DataLoader(image_dsets[x],
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=args.num_workers)
        for x in ['gallery', 'query']}

class_names = image_dsets['query'].classes
use_gpu = torch.cuda.is_available()

gallery_path = image_dsets['gallery'].imgs
query_path = image_dsets['query'].imgs
gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

if args.multi:
    mquery_path = image_dsets['multi-query'].imgs
    mquery_cam, mquery_label = get_id(mquery_path)

#####################################################################
#
# Load the trained model
#
#####################################################################
print('[i] loading the trained model...')
if args.PCB:
    model_structure = PCB(751)
else:
    model_structure = ResNet50(751)
model_path_full = os.path.join(args.model_path, args.model_name)
model = load_model(model_structure, model_path_full, args.epoch)

# Remove the final fc layer and classifier layer
if not args.PCB:
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
else:
    model = PCB_test(model)

model = model.eval()
if use_gpu:
    model = model.cuda()

#####################################################################
#
# Extracting features
#
#####################################################################
gallery_feature = extract_feature(model, dset_loaders['gallery'], args)
query_feature = extract_feature(model, dset_loaders['query'], args)
if args.multi:
    mquery_feature = extract_feature(model, dset_loaders['multi-query'], args)

# Save to Matlab for evaluation
if args.multi:
    result = {
        'mquery_f': mquery_feature.numpy(),
        'mquery_label': mquery_label,
        'mquery_cam': mquery_cam
    }
    fname = args.model_name + args.epoch + 'multi_query.mat'
    scipy.io.savemat(os.path.join('./results', fname), result)
else:
    result = {
        'gallery_f': gallery_feature.numpy(),
        'gallery_label': gallery_label,
        'gallery_cam': gallery_cam,
        'query_f': query_feature.numpy(),
        'query_label': query_label,
        'query_cam': query_cam
    }
    fname = args.model_name + args.epoch + 'single_query.mat'
    scipy.io.savemat(os.path.join('./results', fname), result)
