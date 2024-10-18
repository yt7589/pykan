# 复数神经网络实践
# import argparse
import os
import sys
import argparse
import random
from typing import Dict
from struct import unpack, pack
import mmap
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# onnx
import onnx
import onnxscript
import onnxruntime
from apps.rsp.conf.fmcw_config import FmcwConfig as Fcfg
from apps.rsp.dss.cssr_ds import CssrDs
# KAN
from kan import KAN
from kan.utils import create_dataset
from kan.utils import ex_round

class FmcwCkanApp(object):
    def __init__(self):
        self.name = 'apps.rsp.fmcw_ckan_app.FmcwCkanApp'

    @staticmethod
    def startup(params:Dict = {}) -> None:
        print(f'复数KAN网络应用 v0.0.1')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 128
        # 生成数据集
        train_ds = CssrDs(mode='train')
        print(f'训练集数量：{len(train_ds)};')
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        X, y = None, None
        for batch, (Xi, yi) in enumerate(train_dataloader):
            if X is None:
                X = torch.hstack((Xi.real, Xi.imag))
                y = yi.real.unsqueeze(dim=-1)
            else:
                X = torch.vstack(X, torch.hstack(Xi.real, Xi.imag))
                y = torch.vstack(y, yi.real.unsqueeze(dim=-1))
        X = X.to(device)
        y = y.to(device)
        dataset = {
            'train_input': X,
            'train_label': y,
            'test_input': X,
            'test_label': y
        }
        print(f'##### X: {X.shape}; y: {y.shape}; => {y};')
        test_ds = CssrDs(mode='test')
        print(f'测试集数量：{len(test_ds)};')
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        model = KAN(width=[2048,256,64,16,1], grid=3, k=3, seed=42, device=device)
        # # plot KAN at initialization
        # model(X)
        # model.plot_all(img_fn='./figures/init.png')
        # train the model
        model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
        print(f'^_^ The End! ^_^')