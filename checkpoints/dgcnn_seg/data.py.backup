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

from utils.ply import read_ply

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
    def __init__(self, path, num_points, partition='train'):
        super().__init__()
        self.num_points = num_points
        self.partition = partition
        filenames = glob.glob(path + "/"+partition+"/*.ply")
        clouds = [read_ply(f) for f in filenames]
        points = [np.vstack((cloud['x'], cloud['y'], cloud['z'])).T for cloud in clouds]
        if self.partition == "test":
            labels = [np.zeros(cloud.shape[0],dtype=np.int64) for cloud in self.points]
        else:
            labels = [cloud['class'].astype(np.int64) - 1 for cloud in clouds]
            to_keep = [l >= 0 for l in labels]
            labels = [l[k] for l, k in zip(labels, to_keep)]
            points = [l[k] for l, k in zip(points, to_keep)]
        self.points = []
        self.labels = []
        for p,l in zip(points, labels):
            p, l = split_columns(p, l)
            self.points = self.points + p
            self.labels = self.labels + l
        

    def __getitem__(self, item):
        if self.partition == "train":
            idx = np.random.choice(np.arange(len(self.points[item])), size=self.num_points, replace=True)
            return self.points[item][idx], self.labels[item][idx]

    def __len__(self):
        return len(self.points)

if __name__ == '__main__':
    train = MiniChallenge("data/MiniChallenge", 5)
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)
