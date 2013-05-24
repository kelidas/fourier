import numpy as np
import matplotlib.pyplot as p

from scipy.integrate import trapz

def dfourier(x, y, T1=0, n=3):
    if T1 == 0:
        T1 = x.max() - x.min()
    Omega1 = np.pi * 2. / T1

    def get_a0(x, y, T1):
        return 1. / T1 * trapz(y, x)

    def cos_coeff(x, y, N):
        return 2. / T1 * trapz(y * np.cos(N * x * Omega1), x)

    def sin_coeff(x, y, N):
        return 2. / T1 * trapz(y * np.sin(N * x * Omega1), x)

    N = np.arange(n)[:, None] + 1
    a0 = get_a0(x, y, T1)
    cc = cos_coeff(x, y, N)
    ss = sin_coeff(x, y, N)
    print a0
    print cc
    print ss

    yy = a0 + np.sum(cc[:, None] * np.cos(N * x * Omega1) + ss[:, None] *
                     np.sin(N * x * Omega1), axis=0)

    freq = Omega1 * N / 2. / np.pi

    p.plot(x, y, label='function')
    p.plot(x, yy.T, label='fourier series')
    p.xlabel('time')
    p.ylabel('y')
    p.legend()

    p.figure()
    p.grid()
    p.vlines(N - 0.05, [0], cc, color='blue', label='cos')
    p.vlines(N + 0.05, [0], ss, color='green', label='sin')
    p.xlabel('N')
    p.ylabel('coeff')
    p.legend()

    p.figure()
    p.grid()
    p.vlines(freq, [0], cc, color='blue', label='cos')
    p.vlines(freq, [0], ss, color='green', label='sin')
    p.xlabel('frequency')
    p.ylabel('coeff')
    p.legend()

    p.show()

if __name__ == '__main__':

    # data = np.loadtxt(r'../examples/test.txt', skiprows=1)
    # x = data[:, 0]
    # y = data[:, 1]

    sam = 1000
    # x = np.linspace(-np.pi,np.pi,sam)
    # y = np.ones(sam) * np.abs(x)- np.pi/2.
    x = np.linspace(-np.pi, np.pi, sam)
    y = np.ones(sam) * 10 * (x >= 0) - 5
    y[0] = 0
    y[-1] = 0
    # x = np.linspace(-np.pi,np.pi,sam)
    # y = np.ones(sam) * x
    # x = np.linspace(0,6,sam)
    # y = np.cos(2*np.pi*x)

    dfourier(x, y, n=2, T1=6)
