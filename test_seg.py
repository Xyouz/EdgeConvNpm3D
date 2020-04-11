#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
from data import  MiniChallenge
from model import DGCNNSeg
import numpy as np
from torch.utils.data import DataLoader
import random
from tqdm import tqdm


# Solve numpy RNG seeding issue
MAX_INT = 2**32 - 1
def worker_init_fn(worker_id):                                                          
    np.random.seed(random.randrange(MAX_INT))

def test(args):
    dataset = MiniChallenge("data/MiniChallenge/", args.num_points, partition='test', radius=args.radius)    
    test_loader = DataLoader(dataset, num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=False, drop_last=False, worker_init_fn=worker_init_fn)
   
    
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNNSeg(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    for i in tqdm(range(args.repeat)):
        for data, indices in test_loader:
            with torch.no_grad():
                data = data.to(device)
                data = data.permute(0, 2, 1)
                logits = model(data)
                preds = logits.argmax(dim=1).detach().cpu().numpy()
            dataset.update_labels(indices.numpy(), preds)
            
    dataset.write_cloud(args.out_file)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    # parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
    #                     help='Name of the experiment')
    parser.add_argument('--out_file', type=str, default="checkpoints/cloud.ply", metavar='N',
                        help='Output filename')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--repeat', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--tnet', type=int, default=1, choices=[0, 1],
                        help='add a transformer net in front of the model')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--radius', type=int, default=3, metavar='N',
                    help='Radius for the dataset')
    parser.add_argument('--workers', type=int, default=0, metavar='N',
                    help='Number of workers for dataloading.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    args = parser.parse_args()


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print( 'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')


    test(args)
