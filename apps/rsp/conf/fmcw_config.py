# FMCW的配置信息
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

class FmcwConfig(object):
#     # 全局变量定义
#     # 雷达参数
#     c = 3e8  # 光速
#     f0 = 77e9  # 雷达工作频率
#     B = 300e6  # 调频带宽
#     T = 1e-3  # 调频周期
#     R_max = 500  # 最大识别距离
#     R_min = 10  # 最小识别距离
#     # 发射和接收天线参数
#     N_tx = 2  # 发射天线数量
#     N_rx = 4  # 接收天线数量
#     d = 0.5 * c / f0  # 天线间距（假设为半波长）
#     # 目标参数
#     targets = np.array([[100, 20, 0, 0],  # 距离(m), 速度(m/s), 水平到达角(rad), 竖直到达角(rad)
#                         [200, -10, np.pi/4, np.pi/6],
#                         [300, 5, -np.pi/4, -np.pi/6]])
    
    
    C = 3E8
    f_c = 77E9
    B = 150E6
    T_c = 10E-6
    sr = 1024
    R_min = 50.0
    R_max = 500.0
    R = 150.0
    N = 64
    tau_0 = 2*R / C
    device = 'cuda:0'

    def __init__(self):
        self.name = 'apps.rsp.conf.fmcw_conf.FmcwConf'