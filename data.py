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
import h5py
import numpy as np
from torch.utils.data import Dataset
import random

from utils.ply import read_ply, write_ply
from sklearn.neighbors import KDTree

MAX_INT = 2*32-1

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    print("data ", all_data.shape, "label ", all_label.shape)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def split_columns(points, labels, gridsize=5):
    xmin, ymin, zmin = points.min(axis=0)

    res = {}
    for point, label in zip(points, labels):
        index = tuple(((point[:2] - (xmin,ymin))//gridsize).astype(int))
        try:
            res[index].append((point, label))
        except KeyError:
            res[index] = [(point,label)]
    
    points = [np.array([d[0] for d in data]) for data in res.values()]
    labels = [np.array([d[1] for d in data]) for data in res.values()]
    
    return points, labels

class MiniChallenge(Dataset):
    def __init__(self, path, num_points, partition='train',radius=2.5):
        super().__init__()
        self.num_points = num_points
        self.partition = partition
        self.radius = radius
        filenames = glob.glob(path + "/"+partition+"/*al.ply")
        clouds = [read_ply(f) for f in filenames]
        self.points = [np.vstack((cloud['x'], cloud['y'], cloud['z'],cloud['nx'], cloud['ny'], cloud['nz'])).T for cloud in clouds]
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
        # Solve numpy RNG seeding issue
        np.random.seed(random.randrange(MAX_INT))
        if self.partition == 'train':
            p = item // 6
            c = item % 6
            if c == 3: # No pedestrians on some of the point clouds (assumes glob sort files)
                p = self.pedCloud
            idx = np.random.choice(self.wherelabels[p][c])
            indices = self.trees[p].query_radius(self.points[p][np.newaxis,idx,:2], r=self.radius)
            indices = np.random.choice(indices[0], size=self.num_points)
            return self.points[p][indices], self.labels[p][indices]
        else:
            idx = np.random.randint(len(self.points[0]))
            indices = self.trees[0].query_radius(self.points[0][np.newaxis,idx,:2], r=self.radius)
            indices = np.random.choice(indices[0], size=self.num_points)
            return self.points[0][indices], indices

    def __len__(self):
        return 6 * len(self.points) if self.partition == "train" else 32
    
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
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)
