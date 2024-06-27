from collections import OrderedDict
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torchvision.models.resnet
import torchvision.transforms as T
import torchvision
from scipy import stats
import random
import scipy.io as scio
from torch.cuda.amp import autocast as autocast
# from fightingcv_attention.conv.CondConv import *
from fightingcv_attention.conv.DynamicConv import *
from vqvae import VQVAE
from transformers import ViTForImageClassification ,ViTConfig, ViTModel
from sklearn.linear_model import LinearRegression
class Synthetic_distortion_detector(nn.Module):
    def __init__(self):
        super(Synthetic_distortion_detector, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        self.avgpool = model.avgpool
        self.feature = nn.Sequential(OrderedDict({
            name: layer for name, layer in model.named_children()
            if name not in ['avgpool', 'fc']  # 改1：根据自己的网络模型架构调整。
        }))
        self.fc_1000 = nn.Linear(2048, 1000, bias=True)
        self.fc_24 = nn.Linear(1000, 24, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.bn2048 = nn.BatchNorm1d(2048)
        self.bn1000 = nn.BatchNorm1d(1000)

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(-1, 2048)
        x = self.bn2048(x)
        x = self.relu(x)
        x = self.fc_1000(x)
        x = self.bn1000(x)
        x = self.relu(x)
        x = self.fc_24(x)
        x = self.sigmoid(x)
        return x

class vq_vae(nn.Module):
    def __init__(self):
        super(vq_vae, self).__init__()
        cfg = {
            'display_name': 'ava384',
            'image_shape': (3, 384, 384),
            'in_channels': 3,
            'hidden_channels': 128,
            'res_channels': 64,
            'nb_res_layers': 2,
            'embed_dim': 64,
            'nb_entries': 512,
            'nb_levels': 4,
            'scaling_rates': [4, 2, 2, 2],
            'learning_rate': 1e-4,
            'beta': 0.25,
            'batch_size': 128,
            'mini_batch_size': 128,
            'max_epochs': 66,
        }
        self.net = VQVAE(in_channels=cfg['in_channels'],
                         hidden_channels=cfg["hidden_channels"],
                         embed_dim=cfg["embed_dim"],
                         nb_entries=cfg["nb_entries"],
                         nb_levels=cfg["nb_levels"],
                         scaling_rates=cfg["scaling_rates"])
        vq_path = 'vqvaebest_384.pth'
        self.net.load_state_dict(torch.load(vq_path)['model'])
        #加载你的vq_vae预训练参数
    def forward(self, x):

        y,diffs,e,d,id_output=self.net(x)
        # print(e[0].shape)
        # print(e[1].shape)
        # print(e[2].shape)
        # print(e[3].shape)
        return e[0],e[1],e[2],e[3]


class Mutual_adaptation_net(nn.Module):
    def __init__(self):
        super(Mutual_adaptation_net, self).__init__()
        resnet_model = Synthetic_distortion_detector()
        path = 'prior_384.pth'
        # 加载你预训练合成失真检测器的权重
        resnet_model.load_state_dict(torch.load(path))
        self.res_feature = nn.Sequential(OrderedDict({
            name: layer for name, layer in resnet_model.feature.named_children()
            if name not in ['layer1', 'layer2', 'layer3', 'layer4']  # 改1：根据自己的网络模型架构调整。
        }))
        self.res_layer1 = resnet_model.feature.layer1
        self.res_layer2 = resnet_model.feature.layer2
        self.res_layer3 = resnet_model.feature.layer3
        self.res_layer4 = resnet_model.feature.layer4

        self.vae_feature = vq_vae()
        self.res_fc1000 = resnet_model.fc_1000
        self.res_fc24 = resnet_model.fc_24

        for p in self.parameters():
            p.requires_grad = False

        configuration_4 = ViTConfig(image_size=12, num_channels=128, patch_size=12, num_labels=1024,
                                    num_hidden_layers=2)
        self.vit_4 = ViTForImageClassification(configuration_4)

        configuration_3 = ViTConfig(image_size=24, num_channels=128, patch_size=12, num_labels=1024,
                                    num_hidden_layers=2)
        self.vit_3 = ViTForImageClassification(configuration_3)

        configuration_2 = ViTConfig(image_size=48, num_channels=128, patch_size=16, num_labels=1024,
                                    num_hidden_layers=2)
        self.vit_2 = ViTForImageClassification(configuration_2)

        configuration_1 = ViTConfig(image_size=96, num_channels=128, patch_size=16, num_labels=1024,
                                    num_hidden_layers=2)
        self.vit_1 = ViTForImageClassification(configuration_1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.conv1_1 = nn.Sequential(
            nn.Linear(56, 8),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Linear(56, 17),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(16, 1),
            nn.ReLU(inplace=True),
        )

        self.tol1_res = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.tol2_res = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.tol3_res = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.tol4_res = nn.Sequential(
            nn.Linear(6144, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.lh_dy = Dynamic_conv2d(in_planes=1, out_planes=16, kernel_size=64)
        self.hl_dy = Dynamic_conv2d(in_planes=1, out_planes=16, kernel_size=64)
        self.lh_dy1 = Dynamic_conv2d(in_planes=1, out_planes=16, kernel_size=64)
        self.hl_dy1 = Dynamic_conv2d(in_planes=1, out_planes=16, kernel_size=64)
        self.all_dy = Dynamic_conv2d(in_planes=1, out_planes=1, kernel_size=5)

        self.GELU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU()

    def forward(self, x, vae_x):
        # print(x.shape)
        with torch.no_grad():
            y = self.res_feature(x)
            e1, e2, e3, e4 = self.vae_feature(vae_x)
            l1 = self.res_layer1(y)
            l2 = self.res_layer2(l1)
            l3 = self.res_layer3(l2)
            l4 = self.res_layer4(l3)

        h1 = self.vit_1(e1).logits
        h2 = self.vit_2(e2).logits
        h3 = self.vit_3(e3).logits
        h4 = self.vit_4(e4).logits
        h = torch.cat([h1, h2, h3, h4], 1)

        l1_max = torch.flatten(self.maxpool(l1), 1)
        l1_avg = torch.flatten(self.avgpool(l1), 1)
        l1_min = torch.flatten(-self.maxpool(-l1), 1)
        l1 = torch.cat([l1_max, l1_min, l1_avg], 1)

        l2_avg = torch.flatten(self.avgpool(l2), 1)
        l2_max = torch.flatten(self.maxpool(l2), 1)
        l2_min = torch.flatten(-self.maxpool(-l2), 1)
        l2 = torch.cat([l2_max, l2_min, l2_avg], 1)

        l3_max = torch.flatten(self.maxpool(l3), 1)
        l3_avg = torch.flatten(self.avgpool(l3), 1)
        l3_min = torch.flatten(-self.maxpool(-l3), 1)
        l3 = torch.cat([l3_max, l3_min, l3_avg], 1)

        l4_avg = torch.flatten(self.avgpool(l4), 1)
        l4_max = torch.flatten(self.maxpool(l4), 1)
        l4_min = torch.flatten(-self.maxpool(-l4), 1)
        l4 = torch.cat([l4_max, l4_min, l4_avg], 1)

        k = self.res_fc24(self.res_fc1000(l4_avg))
        k = self.sigmoid(k)

        l1 = self.tol1_res(l1)
        l2 = self.tol2_res(l2)
        l3 = self.tol3_res(l3)
        l4 = self.tol4_res(l4)
        l = torch.cat([l1, l2, l3, l4], 1)

        h = torch.unsqueeze(torch.unsqueeze(h, 1), 1)
        n1, c1, h1, w1 = h.shape
        h = h.reshape(n1, 1, 64, 64)
        l = torch.unsqueeze(torch.unsqueeze(l, 1), 1)
        l = l.reshape(n1, 1, 64, 64)

        lh = self.lh_dy(l, h)
        lh = torch.squeeze(torch.squeeze(lh, 2), 2)
        lh = self.GELU(lh)
        lh1 = self.lh_dy1(l, h)
        lh1 = torch.squeeze(torch.squeeze(lh1, 2), 2)
        lh1 = self.GELU(lh1)
        hl = self.hl_dy(h, l)
        hl = torch.squeeze(torch.squeeze(hl, 2), 2)
        hl1 = self.hl_dy1(h, l)
        hl1 = torch.squeeze(torch.squeeze(hl1, 2), 2)
        hl = self.GELU(hl)
        hl1 = self.GELU(hl1)

        l_1 = torch.cat([lh, hl, k], 1)
        l = torch.cat([lh1, hl1, k], 1)
        y = self.conv1(l)
        y_1 = self.conv1_1(l_1)

        y_all = torch.cat([y, y_1], 1)
        y_all = torch.unsqueeze(torch.unsqueeze(y_all, 1), 1)
        y_all = y_all.reshape(n1, 1, 5, 5)
        x = self.all_dy(y_all, y_all)
        x = torch.squeeze(torch.squeeze(x, 2), 2)
        return x, y_1

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, y):
        #将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        #根据y计算权重
        softmax_attention = self.attention(y)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output
