# Complex Single Static Range Dataset 复数单目标静止距离数据集
import os
from typing import Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from apps.rsp.conf.fmcw_config import FmcwConfig as Fcfg

class CssrDs(Dataset):
    DS_BASE = './apps/rsp/work/datasets/fmcw'
    def __init__(self, num:int = 500, mode:str = 'train'):
        # self.convert_ds()
        self.X = torch.load(f'{CssrDs.DS_BASE}/fmcw_ssr_X_{mode}.pt')
        self.y = torch.load(f'{CssrDs.DS_BASE}/fmcw_ssr_y_{mode}.pt')

    def convert_ds(self) -> None:
        X = []
        with open(f'{CssrDs.DS_BASE}/irpoc_fmcw_X_train.txt', 'r', encoding='utf-8') as xfd:
            for row in xfd:
                row = row.strip()
                arrs = row.split(',')
                xi = np.array([complex(x) for x in arrs], dtype=np.complex64)
                X.append(xi)
        self.X = np.array(X, dtype=np.complex64)
        torch.save(self.X, f'{CssrDs.DS_BASE}/fmcw_cssr_X_train.pt')
        print(f'### self.X: {self.X.shape}; {self.X.dtype};')
        y = []
        with open(f'{CssrDs.DS_BASE}/irpoc_fmcw_y_train.txt', 'r', encoding='utf-8') as yfd:
            for row in yfd:
                row = row.strip()
                yi = np.array([complex(float(row), 0)], dtype=np.complex64)
                y.append(yi)
        self.y = np.array(y, dtype=np.complex64)
        torch.save(self.y, f'{CssrDs.DS_BASE}/fmcw_cssr_y_train.pt')
        print(f'### y: {self.y.shape}; {self.y.dtype};')
        X_t_, y_t_ = [], []
        X_v_, y_v_ = [], []
        for i in range(1):
            idx = random.randint(0, self.X.shape[0])
            X_t_.append(self.X[idx])
            y_t_.append(self.y[idx])
            X_v_.append(self.X[idx])
            y_v_.append(self.y[idx])
        X_t = np.array(X_t_, dtype=np.complex64)
        torch.save(X_t, f'{CssrDs.DS_BASE}/fmcw_cssr_X_test.pt')
        y_t = np.array(y_t_, dtype=np.complex64)
        torch.save(y_t, f'{CssrDs.DS_BASE}/fmcw_cssr_y_test.pt')
        X_v = np.array(X_v_, dtype=np.complex64)
        torch.save(X_v, f'{CssrDs.DS_BASE}/fmcw_cssr_X_val.pt')
        y_v = np.array(y_v_, dtype=np.complex64)
        torch.save(y_v, f'{CssrDs.DS_BASE}/fmcw_cssr_y_val.pt')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def _generate_dataset(num:int, R_min:float, R_max:float) -> Tuple[torch.tensor, torch.tensor]:
        '''
        ToDo: 添加v_main, v_max, theta_min(水平), theta_max, phi_min(竖直), phi_max
        '''
        X = None
        R = (R_min + np.random.rand(num)*R_max)/100.0
        for i in range(num):
            x = []
            for n in range(Fcfg.N):
                ev = np.exp(complex(0,1)*((2*np.pi*(Fcfg.B/Fcfg.T_c)*(2*R[i]/Fcfg.C)*(Fcfg.T_c/Fcfg.N*n)) + \
                                        2*np.pi*Fcfg.f_c*(2*R[i]/Fcfg.C) \
                                            - np.pi*(Fcfg.B/Fcfg.T_c)*(2*R[i]/Fcfg.C)**2)
                )
                x.append(np.float32(np.real(ev)))
                x.append(np.float32(np.imag(ev)))
            if X is None:
                X = torch.tensor(x)
            else:
                X = torch.vstack((X, torch.tensor(x)))
            if i % 100 == 0:
                print(f'生成进度：{i}/{num}')
        y = torch.tensor(R.reshape(R.shape[0], 1), dtype=torch.float32)
        return X, y
    
    

    @staticmethod
    def t001(num:int, R_min:float, R_max:float) -> Tuple[torch.tensor, torch.tensor]:
        '''
        ToDo: 添加v_main, v_max, theta_min(水平), theta_max, phi_min(竖直), phi_max
        '''
        print(f'step 2')
        X = None
        R = [150.0 / 100.0] #(R_min + np.random.rand(num)*R_max)/100.0
        for i in range(num):
            x = []
            for n in range(Fcfg.N):
                ev = np.exp(complex(0,1)*((2*np.pi*(Fcfg.B/Fcfg.T_c)*(2*R[i]/Fcfg.C)*(Fcfg.T_c/Fcfg.N*n)) + \
                                        2*np.pi*Fcfg.f_c*(2*R[i]/Fcfg.C) \
                                            - np.pi*(Fcfg.B/Fcfg.T_c)*(2*R[i]/Fcfg.C)**2)
                )
                print(f'### {np.real(ev)}, {np.imag(ev)}; ???????????')
                exit(1)
                x.append(np.float32(np.real(ev)))
                x.append(np.float32(np.imag(ev)))
            if X is None:
                X = torch.tensor(x)
            else:
                X = torch.vstack((X, torch.tensor(x)))
            if i % 100 == 0:
                print(f'生成进度：{i}/{num}')
        y = torch.tensor(R.reshape(R.shape[0], 1), dtype=torch.float32)
        return X, y

# ds = FmcwCmlpDs()