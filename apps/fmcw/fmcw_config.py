#

class FmcwConfig(object):
    C = 3E8
    f_c = 77E9
    B = 150E6
    T_c = 10E-6
    sr = 1024
    R = 150.0
    N = 512
    tau_0 = 2*R / C
    device = 'cpu'