#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import numpy as np
from torch.utils.data import Dataset
import random

from utils.ply import read_ply, write_ply
from sklearn.neighbors import KDTree

MAX_INT = 2*32-1


# Point cloud data augmentation
def translate_pointcloud(pointcloud):
    """
    Apply a translation in the XY plane to the pointcloud.
    """
    xyz = np.random.uniform(low=-10, high=10, size=[2])
       
    translated_pointcloud = pointcloud
    translated_pointcloud[:,:2] += xyz
    return translated_pointcloud

def rotate_pointcloud(pointcloud):
    """
    Apply a rotation around the Z axis to the pointcloud.
    """    
    thetaXY = 2 * np.pi * np.random.rand()
    matXY = np.array([[np.cos(thetaXY), -np.sin(thetaXY)],[np.sin(thetaXY), np.cos(thetaXY)]])
    
    pointcloud[:,:2] = pointcloud[:,:2] @ matXY
    pointcloud[:,3:5] =pointcloud[:,3:5] @ matXY
    return pointcloud

class MiniChallenge(Dataset):
    def __init__(self, path, num_points, partition='train',radius=2.5):
        super().__init__()
        self.num_points = num_points
        self.partition = partition
        self.radius = radius
        filenames = glob.glob(path + "/"+partition+"/*al.ply")
        if len(filenames) == 0:
            print("Warning: the dataloader only reads files ending in 'al.ply'.")
        clouds = [read_ply(f) for f in filenames]
        self.points = [np.vstack((cloud['x'], cloud['y'], cloud['z'],cloud['nx'], cloud['ny'], cloud['nz'])).T for cloud in clouds]
        self.means = [points.mean(axis=0)[:2] for points in self.points]
        if self.partition == "test":
            self.labels = [np.zeros(cloud.shape[0],dtype=np.int32) for cloud in self.points]
        else:
            self.labels = [cloud['scalar_class'].astype(np.int64) - 1 for cloud in clouds]
            to_keep = [l >= 0 for l in self.labels]
            self.labels = [l[k] for l, k in zip(self.labels, to_keep)]
            self.points = [l[k] for l, k in zip(self.points, to_keep)]
            self.wherelabels = [[np.nonzero(l == c)[0] for c in range(6)] for l in self.labels]
            # Only one of the clouds has pedestrians
            self.pedCloud = max(range(len(clouds)), key=lambda i: self.wherelabels[i][3].shape)

        self.trees = [KDTree(p[:,:2]) for p in self.points]


    def __getitem__(self, item):
        if self.partition == 'train':
            p = item // 5
            c = item % 5
            if c >= 3: # No pedestrians on some of the point clouds
                c += 1
            if item == 15: 
                p = self.pedCloud
                c = 3
            idx = np.random.choice(self.wherelabels[p][c])
            indices = self.trees[p].query_radius(self.points[p][np.newaxis,idx,:2], r=self.radius)
            indices = np.random.choice(indices[0], size=self.num_points)
            points = self.points[p][indices]
            points[:,:2] -= self.means[p]
            return translate_pointcloud(rotate_pointcloud(points)), self.labels[p][indices]
        else:
            idx = np.random.randint(len(self.points[0]))
            indices = self.trees[0].query_radius(self.points[0][np.newaxis,idx,:2], r=self.radius)
            indices = np.random.choice(indices[0], size=self.num_points)
            return self.points[0][indices], indices

    def __len__(self):
        return 5 * len(self.points) + 1 if self.partition == "train" else 32
    
    def update_labels(self, indices, labels):
        labels = labels.reshape(-1)
        indices = indices.reshape(-1)
        self.labels[0][indices] = labels + 1
    
    def write_cloud(self, path):
        points = self.points[0]
        labels = self.labels[0]
        write_ply(path, [points[:,:3], labels] , ['x','y','z', 'class'] )


if __name__ == '__main__':
    train = MiniChallenge("data/MiniChallenge", 5)

