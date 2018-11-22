# ----------------------------------------------------------
#
# Project name: DMCRID-MulCAM-reID
# File name: utils.py
#
# Last modified: 2018-11-21
#
# ----------------------------------------------------------


# Import the necessary packages
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

from shutil import copyfile
from torch.autograd import Variable

matplotlib.use('agg')


# -------------------------------------------------------------------
#
# Helper functions
#
# -------------------------------------------------------------------

def split_subfolder(base_path, folder_name, save_path, save_folder_name):
    folder_path = os.path.join(base_path, folder_name)
    saved_folder_path = os.path.join(save_path, save_folder_name)

    if not os.path.exists(saved_folder_path):
        os.makedirs(saved_folder_path)

    for root, dirs, files in os.walk(folder_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = folder_path + '/' + name
            dst_path = saved_folder_path + '/' + ID[0]

            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            copyfile(src_path, dst_path + '/' + name)


def split_train_val(base_path, folder_name, save_path, train_name, val_name):
    folder_path = os.path.join(base_path, folder_name)
    train_path = os.path.join(save_path, train_name)
    val_path = os.path.join(save_path, val_name)

    if not os.path.exists(train_path):
        os.makedirs(train_path)
        os.makedirs(val_path)

    for root, dirs, files in os.walk(folder_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = folder_path + '/' + name
            dst_path = train_path + '/' + ID[0]

            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
                # The first image is used as val image
                dst_path = val_path + '/' + ID[0]
                os.makedirs(dst_path)

            copyfile(src_path, dst_path + '/' + name)


# --------------------------------------------------------------------
#
# Draw Curved Line
#
# --------------------------------------------------------------------
epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title='loss')
ax1 = fig.add_subplot(122, title='top1-err')


def draw_curve(current_epoch, y_loss, y_error, model_name):
    """
    Draw the curved line of the training phase

    :param current_epoch: The current epoch
    :param y_loss: y_loss
    :param y_error: y_error
    :param name: The name of model
    :return: None
    """

    epoch.append(current_epoch)
    ax0.plot(epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(epoch, y_error['train'], 'bo-', label='train')
    ax1.plot(epoch, y_error['val'], 'ro-', label='val')

    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', model_name, 'curved_line.jpg'))


# --------------------------------------------------------------------
#
# Save model
#
# --------------------------------------------------------------------
def save_model(network, epoch_label, model_path):
    save_filename = 'net_{}.pth'.format(epoch_label)
    save_path = os.path.join(model_path, save_filename)
    torch.save(network.cpu().state_dict(), save_path)


# --------------------------------------------------------------------
#
# Load the trained model
#
# --------------------------------------------------------------------
def load_model(network, model_path, epoch):
    """
    Load the trained model which specific epoch
    :param network: The model trained in training
    :param model_path: Path to the model
    :param epoch: which epoch
    :return: The trained model
    """

    saved_path = os.path.join(model_path, 'net_{}.pth'.format(epoch))
    network.load_state_dict(torch.load(saved_path))

    return network


def fliplr(img):
    """
    Flip horizontal
    :param img: The input image
    :return: The flipped image
    """

    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    flipped_img = img.index_select(3, inv_idx)

    return flipped_img


def get_id(img_path):
    """
    Get id of the input images
    :param img_path: Path to the images
    :return: camera_id, labels
    """

    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))

    return camera_id, labels


# --------------------------------------------------------------------
#
# Extract feature
#
# --------------------------------------------------------------------
def extract_feature(model, data_loaders, args):
    """
    Extract feature from  a trained model
    :param model:
    :param data_loaders:
    :return:
    """

    features = torch.FloatTensor()
    count = 0
    for data in data_loaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print('{}-th image'.format(count))
        if args.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_()
        else:
            ff = torch.FloatTensor(n, 2048).zero_()
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff + f

        #####################################################################
        #
        # normalize features
        #
        if args.PCB:
            # feature size (n, 2048, 6)
            # 1. To treat every part equally, I calculate the norm for every
            # 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to normalize
            # the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)

    return features


# -------------------------------------------------------------------
#
# Test split_subfolder
#
# -------------------------------------------------------------------
if __name__ == '__main__':
    download_path = './datasets/Market-1501'
    if not os.path.exists(download_path):
        print('[!] please change the download_path')

    save_path = os.path.join(download_path, 'demo')
    print(save_path)
    # ./datasets/Market-1501/demo


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    split_subfolder(download_path, 'query', save_path, 'query')
