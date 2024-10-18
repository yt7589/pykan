# 探索使用KAN来拟合FFT
import argparse
import matplotlib.pyplot as plt
import torch
from kan import KAN
from kan.utils import create_dataset
from kan.utils import ex_round

def main(args:argparse = {}) -> None:
    print(f'CKAN v0.0.1')
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
    model = KAN(width=[2,5,1], grid=3, k=3, seed=42, device=device)
    # create dataset f(x,y) = exp(sin(pi*x)+y^2)
    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(f, n_var=2, device=device)
    dataset['train_input'].shape, dataset['train_label'].shape
    # plot KAN at initialization
    model(dataset['train_input'])
    model.plot()
    plt.show()
    # train the model
    model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
    model.plot()
    plt.show()
    # Prune the model
    model = model.prune()
    model.plot()
    plt.show()
    # refine
    model.fit(dataset, opt="LBFGS", steps=50)
    model = model.refine(10)
    model.fit(dataset, opt="LBFGS", steps=50)


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
    model.fit(dataset, opt="LBFGS", steps=50)
    formula = ex_round(model.symbolic_formula()[0][0],4)
    print(f'formula: {formula};')


    print(f'^_^ The End! ^_^')

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_mode', action='store',
        type=int, default=1, dest='run_mode',
        help='run mode'
    )
    return parser.parse_args()

# v1
if '__main__' == __name__:
    args = parse_arg()
    main(args=args)