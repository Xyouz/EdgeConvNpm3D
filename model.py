#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

class EdgeConv(nn.Module):
    def __init__(self, input_channels, output_channels, k):
        super().__init__()
        self.k = k
        self.input_channels = input_channels
        if type(output_channels) == int:
            self.output_channels = [output_channels]
        else:
            self.output_channels = output_channels
        
        layers = [nn.Conv2d(2*self.input_channels, self.output_channels[0],
                                kernel_size=1, bias=False),
                nn.BatchNorm2d(self.output_channels[0]),
                nn.LeakyReLU(negative_slope=0.2)
                ]

        outp = self.output_channels[0]
        for out in self.output_channels[1:]:
            layers.append(nn.Conv2d(outp, out, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            outp = out

        self.layers = nn.Sequential(*layers)

    
    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.layers(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x

class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        

        self.edge1 = EdgeConv(3, 64, self.k)
        self.edge2 = EdgeConv(64, 64, self.k)
        self.edge3 = EdgeConv(64, 128, self.k)
        self.edge4 = EdgeConv(128, 256, self.k)

        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.edge1(x)
        x2 = self.edge2(x1)
        x3 = self.edge3(x2)
        x4 = self.edge4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class TransformNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cpu" if self.args.no_cuda else "cuda")
        self.k = self.args.k
        self.num_points = self.args.num_points
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.conv1 = nn.Conv2d(12,64, kernel_size=1, bias=False)
        self.lkrl1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(64,128, kernel_size=1, bias=False)
        self.lkrl2 = nn.LeakyReLU(negative_slope=0.2)
        self.maxpool1 = nn.MaxPool2d((1,self.k))

        self.conv3 = nn.Conv1d(128,1024, kernel_size=1, bias=False)
        self.lkrl3 = nn.LeakyReLU(negative_slope=0.2)
        self.maxpool2 = nn.MaxPool1d(self.num_points)

        self.fc1 = nn.Linear(1024,512,bias=False)
        self.lkrl4 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(512,256,bias=False)
        self.lkrl5 = nn.LeakyReLU(negative_slope=0.2)

        self.fc3 = nn.Linear(256,9)
        self.fc4 = nn.Linear(256,3)

    def forward(self, input):
        x = get_graph_feature(input, k=self.k)
        x = self.lkrl1(self.bn1(self.conv1(x)))
        x = self.lkrl2(self.bn2(self.conv2(x)))
        x = self.maxpool1(x)

        x = torch.squeeze(x)
        x = self.lkrl3(self.bn3(self.conv3(x)))
        x = self.maxpool2(x)
        x = torch.squeeze(x)

        x = self.lkrl4(self.bn4(self.fc1(x)))
        x = self.lkrl5(self.bn5(self.fc2(x)))

        bias = self.fc4(x).view(-1,1,3)
        matrix = torch.eye(3, device=self.device) + self.fc3(x).view(-1,3,3)
        
        mul = torch.bmm(input.transpose(2,1), matrix)
        return (mul + bias).transpose(2,1)




class DGCNNSeg(nn.Module):
    def __init__(self, args, output_channels=6):
        super().__init__()
        self.args = args
        self.k = args.k

        #self.STN = TransformNet(self.args)

        self.edge1 = EdgeConv(6,[64,64],self.k)
        self.edge2 = EdgeConv(64,[64,64],self.k)
        self.edge3 = EdgeConv(64,64,self.k)

        self.bn4 = nn.BatchNorm1d(args.emb_dims)
        self.conv4 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn5 = nn.BatchNorm1d(256)
        self.conv5 = nn.Sequential(nn.Conv1d(args.emb_dims + 3*64, 256, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn6 = nn.BatchNorm1d(256)
        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn7 = nn.BatchNorm1d(128)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.dp1 = nn.Dropout(self.args.dropout)
        self.conv8 = nn.Conv1d(128, output_channels, kernel_size=1, bias=False)
        



        

    def forward(self, x):
        #x = self.STN(x)
        
        x1 = self.edge1(x)
        x2 = self.edge2(x1)

        x3 = self.edge3(x2)
        x123 = torch.cat((x1,x2,x3), dim=1)     

        x = self.conv4(x123)
        x = x.max(dim=-1, keepdim=True)[0] 

        x = x.repeat(1, 1, self.args.num_points)
        x = torch.cat((x, x123), dim=1)

        x = self.conv5(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.dp1(x)
        x = self.conv8(x)

        return x
        