#
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import sympy as sp
from sympy.abc import x
from kan.utils import create_dataset
from kan import KAN

class Exp002(object):
    def __init__(self):
        self.name = 'apps.exp.exp002.Exp002'

    @staticmethod
    def startup(args:argparse.Namespace = {}) -> None:
        print(f'示例2')
        # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
        model = KAN(width=[4,2,1,1], grid=3, k=3, seed=0)
        f = lambda x: torch.exp((torch.sin(torch.pi*(x[:,[0]]**2+x[:,[1]]**2))+torch.sin(torch.pi*(x[:,[2]]**2+x[:,[3]]**2)))/2)
        dataset = create_dataset(f, n_var=4, train_num=3000)
        print(f'dataset: {type(dataset)}; {type(dataset["train_input"])}; {dataset["train_input"].shape};')
        # train the model
        model.train(dataset, opt="LBFGS", steps=20, lamb=0.001, lamb_entropy=2.)