#
from typing import Dict
import torch
from kan import KAN
from kan.utils import create_dataset
from kan.utils import ex_round

class KanApp(object):
    def __init__(self):
        self.name = 'apps.rsp.kan_app.KanApp'

    def startup(self, args:Dict = {}) -> None:
        torch.set_default_dtype(torch.float64)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        print(f'current device: {device};')
        # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
        model = KAN(width=[2,5,1], grid=3, k=3, seed=42, device=device)
        print(f'model: {model};')
        # create dataset f(x,y) = exp(sin(pi*x)+y^2)
        f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        dataset = create_dataset(f, n_var=2, device=device)
        print(f'dataset: {type(dataset)};')
        print(f'X: {dataset["train_input"].shape}; {type(dataset["train_input"])}; Y:{dataset["train_label"].shape}, {type(dataset["train_label"])};')
        exit(0)
        # plot KAN at initialization
        model(dataset['train_input'])
        model.plot_all(img_fn='./figures/init.png')
        # train the model
        model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
        model.plot_all(img_fn='./figures/train_0.png')
        # prune
        model = model.prune()
        model.plot_all(img_fn='./figures/prune_0.png')
        # continue training
        model.fit(dataset, opt="LBFGS", steps=50)
        model = model.refine(10)
        model.fit(dataset, opt="LBFGS", steps=50)
        # symbolic regression
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
        model.fit(dataset, opt="LBFGS", steps=50)
        formula = ex_round(model.symbolic_formula()[0][0],4)
        print(f'formula: {formula};')
