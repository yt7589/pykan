# Radar Signal Process Main Entry Point
import argparse
from apps.rsp.simu.fmcw_simu import FmcwSimu
from apps.rsp.kan_app import KanApp
from apps.rsp.fmcw_cmlp_app import FmcwCmlpApp
from apps.rsp.fmcw_ckan_app import FmcwCkanApp

def main(args:argparse.Namespace = {}) -> None:
    print(f'KAN雷达信号处理端到端应用 v0.0.1')
    # simu = FmcwSimu()
    # simu.startup()
    params = vars(args)
    if 1 == args.run_mode: # KAN网络尝试
        app = KanApp()
        app.startup()
    elif 2 == args.run_mode: # 复数MLP
        FmcwCmlpApp.startup(params=params)
    elif 3 == args.run_mode: # 复数KAN
        FmcwCkanApp.startup(params=params)

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