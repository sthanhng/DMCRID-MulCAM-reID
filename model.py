# ----------------------------------------------------------
#
# Project name: DMCRID-MulCAM-reID
# File name: model.py
#
# Last modified: 2018-11-21
#
# ----------------------------------------------------------


import torch
import torch.nn as nn

from torch.nn import init
from torchvision import models


######################################################################


def weights_init_kaiming(m):
    """

    :param m:
            None
    :return:
            None
    """

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    """

    :param m:
            None
    :return:
            None
    """

    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


###############################################################################


# -------------------------------------------------------------------
#
# Defines the new fully connected layer and classification layer
#
# |          |   |      |   |        |   |          |
# |  Linear  |-->|  BN  |-->|  ReLu  |-->|  Linear  |
# |          |   |      |   |        |   |          |
#
# -------------------------------------------------------------------
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True,
                 num_bottleneck=512):
        """

        :param input_dim:

        :param class_num:

        :param dropout:

        :param relu:

        :param num_bottleneck:

        """

        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


# -------------------------------------------------------------------
#
# Define the ResNet50-based Model
#
# -------------------------------------------------------------------
class ResNet50(nn.Module):
    def __init__(self, class_num):
        """

        :param class_num: Number of classes

        """

        super(ResNet50, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


# -------------------------------------------------------------------
#
# Define the Beyond Part Model-based model
# @Beyond Part Models: Person Retrieval with Refined Part Pooling
# Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, Shengjin Wang (2018)
#
# -------------------------------------------------------------------
class PCB(nn.Module):
    def __init__(self, class_num):
        """

        :param class_num: Number of classes

        """

        super(PCB, self).__init__()

        self.part = 6  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batch-size*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


# -------------------------------------------------------------------
#
# Define the PCB model for testing
#
# -------------------------------------------------------------------
class PCB_test(nn.Module):
    def __init__(self, model):
        """

        :param model: The model

        """

        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y
