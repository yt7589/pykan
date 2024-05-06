#
import argparse
import matplotlib
import matplotlib.pyplot as plt
import torch
import sympy as sp
from sympy.abc import x
from kan.utils import create_dataset
from kan import KAN
from apps.exp.exp001 import Exp001
from apps.exp.exp002 import Exp002
from apps.fmcw.fmcw_app import FmcwApp

class ExpApp(object):
    def __init__(self):
        self.name = 'apps.exp.exp_app.ExpApp'

    @staticmethod
    def startup(args:argparse.Namespace = {}) -> None:
        print(f'KAN测试程序 v0.0.2')
        if 0 == args.run_mode:
            ExpApp.hello_ipynb(args=args)
        elif 1 == args.run_mode:
            Exp001.startup(args=args)
        elif 2 == args.run_mode:
            Exp002.startup(args=args)
        elif 3 == args.run_mode:
            FmcwApp.startup(args=args)

    @staticmethod
    def hello_ipynb(args:argparse.Namespace = {}) -> None:
        # 指定字体名称
        matplotlib.rcParams['font.family'] = 'SimHei' 
        matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        # [1] create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
        model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        # [2] create dataset f(x,y) = exp(sin(pi*x)+y^2)
        f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        dataset = create_dataset(f, n_var=2)
        print(f'X: {dataset["train_input"].shape}, y: {dataset["train_label"].shape}')
        # [3] plot KAN at initialization
        model(dataset['train_input'])
        model.plot(beta=100)
        plt.title(f'网络初始化状态')
        plt.show()
        # [4] [5] train the model
        model.train(dataset, opt="LBFGS", steps=30, lamb=0.01, lamb_entropy=10.)
        model.plot()
        plt.title(f'训练后的网络')
        plt.show()
        # [6] prune the model
        model.prune(threshold=5e-2)
        model.plot(mask=True)
        plt.title(f'剪枝后的网络')
        plt.show()
        # [7] Simplify the model
        model = model.prune(threshold=1e-2)
        model(dataset['train_input'])
        model.plot()
        plt.title(f'经过化简后的网络')
        plt.show()
        # [8][9] continue train and plot
        model.train(dataset, opt="LBFGS", steps=50)
        model.plot()
        plt.title(f'继续训练后的网络')
        plt.show()
        # [10] Automatically or manually set activation functions to be symbolic
        mode = "auto" # "manual"
        if mode == "manual":
            # manual mode
            model.fix_symbolic(0,0,0,'sin')
            model.fix_symbolic(0,1,0,'x^2')
            model.fix_symbolic(1,0,0,'exp')
        elif mode == "auto":
            # automatic mode
            lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
            model.auto_symbolic(lib=lib)
        # [11] train the net
        model.train(dataset, opt="LBFGS", steps=50)
        # [12] 获取公式
        rst = model.symbolic_formula()
        print(f'### rst: {type(rst[0][0])};')
        latex_expr = sp.latex(rst[0][0])
        print(latex_expr)
        print(f'^_^ The End! ^_^')

def main(args:argparse.Namespace = {}) -> None:
    ExpApp.startup(args=args)

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