#
from typing import Tuple
import numpy as np
import torch
from apps.fmcw.fmcw_config import FmcwConfig as Fcfg

class FmcwDs(object):
    def __init__(self):
        self.name = 'apps.fmcw.fmcw_ds.FmcwDs'

    @staticmethod
    def create_dataset():
        dataset = {}
        train_num = 5000
        test_num = 1000
        R_min, R_max = 50.0, 500.0
        np.random.seed(100)
        dataset['train_input'], dataset['train_label'] = FmcwDs.generate_dataset(train_num, R_min=R_min, R_max=R_max)
        dataset['test_input'], dataset['test_label'] = FmcwDs.generate_dataset(test_num, R_min=R_min, R_max=R_max)
        return dataset

    @staticmethod
    def generate_dataset(num:int, R_min:float, R_max:float) -> Tuple[torch.tensor, torch.tensor]:
        '''
        ToDo: 添加v_main, v_max, theta_min(水平), theta_max, phi_min(竖直), phi_max
        '''
        X = None
        R = R_min + np.random.rand(num)*R_max
        for i in range(num):
            x = []
            for n in range(Fcfg.N):
                ev = np.exp(complex(0,1)*((2*np.pi*(Fcfg.B/Fcfg.T_c)*(2*R[i]/Fcfg.C)*(Fcfg.T_c/Fcfg.N*n)) + \
                                        2*np.pi*Fcfg.f_c*(2*R[i]/Fcfg.C) \
                                            - np.pi*(Fcfg.B/Fcfg.T_c)*(2*R[i]/Fcfg.C)**2))
                x.append(ev)
            if X is None:
                X = np.array(x)
            else:
                X = np.vstack((X, np.array(x)))
            if i % 100 == 0:
                print(f'生成进度：{i}/{num}')
        return torch.tensor(X).to(Fcfg.device), torch.tensor(R).to(Fcfg.device)
            