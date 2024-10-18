# KAN用于FFT开发手册

# 1. 问题描述
我们研究的问题为：
$$
y = \rho A e^{j2\pi f \frac{2R}{C}}=\rho A \Big( \cos (2\pi f \frac{2R}{C}) + j \sin(2\pi f \frac{2R}{C}) \Big)
$$
其中$A$、$f$、$C$为已知变量，而$\rho$、$R$为未知变量。我们可以从数据中观察到一系列

# A. 附录
## A.1. 环境搭建
```bash
conda activate rsp # torch=2.4 python=3.11
# 安装依赖库
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn matplotlib pyyaml tqdm pandas 
```