# 调频连续波雷达应用
import argparse
import matplotlib.pyplot as plt
import torch
import sympy as sp
from sympy.abc import x
from kan import KAN
from apps.fmcw.fmcw_ds import FmcwDs



class FmcwApp(object):
    def __init__(self):
        self.name = 'apps.fmcw.fmcw_app.FmcwApp'

    @staticmethod
    def startup(args:argparse.Namespace = {}) -> None:
        print(f'调频连续波KAN应用 v0.0.2')
        dataset = FmcwDs.create_dataset()
        print(f'X_train: {dataset["train_input"].shape}; y_train: {dataset["train_label"].shape};')
        print(f'X_test: {dataset["test_input"].shape}; y_test: {dataset["test_label"].shape};')
        model = KAN(width=[512,256,128,64,32,16,1], grid=3, k=3, seed=0)
        model.train(dataset, opt="LBFGS", steps=20, lamb=0.001, lamb_entropy=2.)

def main(args:argparse.Namespace = {}) -> None:
    FmcwApp.startup(args=args)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_mode', action='store',
        type=int, default=1, dest='run_mode',
        help='run mode'
    )
    return parser.parse_args()

if '__main__' == __name__:
    args = parse_args()
    main(args=args)