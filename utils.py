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

from shutil import copyfile

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
