# -*- coding: utf-8 -*-
# @Time:     7/5/2021 2:39 PM
# @Author:   Zihao Wang, zwang@mpi-magdeburg.mpg.de
# @File:     utils.py


import torch
from sklearn.preprocessing import StandardScaler


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.update = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.update = True
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.update = True
        elif self.best_loss - val_loss < self.min_delta:
            self.update = False
            self.counter += 1
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def y_scaling(dataset):
    items = dataset.items
    y = torch.Tensor([item.y for item in items])
    scaler = StandardScaler()
    scaler.fit(y)
    print(scaler.mean_, scaler.var_)
    for item in items:
        item.y = scaler.transform(item.y.reshape(1, -1))[0]
    return dataset, scaler


def struc_scaling(dataset):
    items = dataset.items
    y = torch.Tensor([item.struc for item in items])
    scaler = StandardScaler()
    scaler.fit(y)
    for item in items:
        item.struc = scaler.transform(item.struc.reshape(1, -1))[0]
    return dataset, scaler


def items2data(items):
    names, mof_x, mof_edge_index, mof_node, mof_topo, y = [], [], [], [], [], []
    for item in items:
        names.append(item.name)
        mof_x.append(item.x)
        mof_edge_index.append(item.edge_index)
        mof_node.append(item.node)
        mof_topo.append(item.topo)
        y.append(item.y)
    y = torch.tensor(y, dtype=torch.float32)
    return names, mof_x, mof_edge_index, mof_node, mof_topo, y
