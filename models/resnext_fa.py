import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

import models.fa_module

from models.fa_module import falayer

import pdb

__all__ = ['ResNeXt', 'resnet50', 'resnet101']

# class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None, filter_size = 28):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
# class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
# class torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
# class torch.nn.ReLU(inplace=False)
        self.relu = nn.ReLU(inplace=True)
        #pdb.set_trace();
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNeXtBottleneck_fa(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None, filter_size = 28):
        super(ResNeXtBottleneck_fa, self).__init__()
        mid_planes = cardinality * int(planes / 32)
# class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
# class torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
# class torch.nn.ReLU(inplace=False)
        self.relu = nn.ReLU(inplace=True)
        #pdb.set_trace();
        self.fa = falayer(planes * 2, filter_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        pdb.set_trace();

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        pdb.set_trace();

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        pdb.set_trace();

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.fa(out)
        pdb.set_trace();

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

# [3, 4, 23, 3]
    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
# class torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True)
        self.bn1 = nn.BatchNorm3d(64)
# class torch.nn.ReLU(inplace=False)
        self.relu = nn.ReLU(inplace=True)
# class torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
# _make_layer方法的第一个输入block是Bottleneck或BasicBlock类，第二个输入是该blocks的输出channel，第三个输入是每个blocks中包含多少个residual子结构，因此layers这个列表就是前面resnet101的[3, 4, 23, 3]
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))
# *arg表示任意多个无名参数,类型为tuple;**kwargs表示关键字参数,为dict。
# python 里面列表前面加星号, 作用是将列表解开成两个独立的参数，传入函数，还有类似的有两个星号，是将字典解开成独立的元素作为形参。
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNeXt_fa(nn.Module):

# [3, 4, 23, 3]
    def __init__(self,
                 block,
                 block_fa,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400):
        self.inplanes = 64
        super(ResNeXt_fa, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
# class torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True)
        self.bn1 = nn.BatchNorm3d(64)
# class torch.nn.ReLU(inplace=False)
        self.relu = nn.ReLU(inplace=True)
# class torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
# _make_layer方法的第一个输入block是Bottleneck或BasicBlock类，第二个输入是该blocks的输出channel，第三个输入是每个blocks中包含多少个residual子结构，因此layers这个列表就是前面resnet101的[3, 4, 23, 3]
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality, img_size = 28)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2,
            img_size = 14)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2,
            img_size = 7)
        self.layer4 = self._make_layer(
            block_fa, 1024, layers[3], shortcut_type, cardinality, stride=2, 
            img_size = 4)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1,
                    img_size = 28):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
#    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample, filter_size = img_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality, filter_size = img_size))
# *arg表示任意多个无名参数,类型为tuple;**kwargs表示关键字参数,为dict。
# python 里面列表前面加星号, 作用是将列表解开成两个独立的参数，传入函数，还有类似的有两个星号，是将字典解开成独立的元素作为形参。
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        pdb.set_trace()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #pdb.set_trace()
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def get_fine_tuning_parameters(model, ft_begin_index):
    #pdb.set_trace();
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

def get_fine_tuning_parameters_fa(model, base_lr):
    print("Here in get_fine_tuning_parameters_fa");
    parameters = []
    for k, v in model.named_parameters():
        if ("_fa" in k):
            print("%s" %k);
            parameters.append({'params': v, 'lr': (10 * base_lr)});
        else:
            parameters.append({'params': v, 'lr': base_lr});
    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
# *arg表示任意多个无名参数,类型为tuple;**kwargs表示关键字参数,为dict。
    model = ResNeXt_fa(ResNeXtBottleneck, ResNeXtBottleneck_fa, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
