#
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import sympy as sp
from sympy.abc import x
from kan.utils import create_dataset
from kan import KAN

class Exp001(object):
    def __init__(self):
        self.name = 'apps.exp.exp001.Exp001'

    @staticmethod
    def startup(args:argparse.Namespace = {}) -> None:
        print(f'KAN例程1')
        # 1. intialize model and create dataset
        # initialize KAN with G=3
        model = KAN(width=[2,1,1], grid=3, k=3)
        # create dataset
        f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        dataset = create_dataset(f, n_var=2)
        # 2. train
        model.train(dataset, opt="LBFGS", steps=20)
        # 3. The loss plateaus. we want a more fine-grained KAN!
        # initialize a more fine-grained KAN with G=10
        model2 = KAN(width=[2,1,1], grid=10, k=3)
        # initialize model2 from model
        model2.initialize_from_another_model(model, dataset['train_input'])
        # 4. Train model2
        model2.train(dataset, opt="LBFGS", steps=20)
        # 5. Now we can even iteratively making grids finer.
        grids = np.array([5,10,20,50,100])
        train_losses = []
        test_losses = []
        steps = 50
        k = 3
        for i in range(grids.shape[0]):
            if i == 0:
                model = KAN(width=[2,1,1], grid=grids[i], k=k)
            if i != 0:
                model = KAN(width=[2,1,1], grid=grids[i], k=k).initialize_from_another_model(model, dataset['train_input'])
            results = model.train(dataset, opt="LBFGS", steps=steps, stop_grid_update_step=30)
            train_losses += results['train_loss']
            test_losses += results['test_loss']
        # 6. Training dynamics of losses display staircase structures (loss suddenly drops after grid refinement)
        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.legend(['train', 'test'])
        plt.ylabel('RMSE')
        plt.xlabel('step')
        plt.yscale('log')
        plt.show()
        # 7. Neural scaling laws
        n_params = 3 * grids
        train_vs_G = train_losses[(steps-1)::steps]
        test_vs_G = test_losses[(steps-1)::steps]
        plt.plot(n_params, train_vs_G, marker="o")
        plt.plot(n_params, test_vs_G, marker="o")
        plt.plot(n_params, 100*n_params**(-4.), ls="--", color="black")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(['train', 'test', r'$N^{-4}$'])
        plt.xlabel('number of params')
        plt.ylabel('RMSE')
        plt.show()
        # 求出符号表示
        mode = "auto" # "manual"
        if mode == "manual":
            # manual mode
            model.fix_symbolic(0,0,0,'sin');
            model.fix_symbolic(0,1,0,'x^2');
            model.fix_symbolic(1,0,0,'exp');
        elif mode == "auto":
            # automatic mode
            lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
            model.auto_symbolic(lib=lib)
        model.train(dataset, opt="LBFGS", steps=50)
        rst = model.symbolic_formula()
        print(f'### rst: {type(rst[0][0])};')
        latex_expr = sp.latex(rst[0][0])
        print(latex_expr)
        print(f'^_^ The End! ^_^')