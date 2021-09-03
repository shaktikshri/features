
def dist(X):
    phi = (1 / np.power(2 * 22 / 7, 0.5)) * torch.exp(-0.5 * (X).pow(2))
    gamma = (1 / 6) * (X + 2).pow(3) * (-1 * (X + 2)).exp()
    mask1 = torch.where(X > -2, 0.3, 1.0)
    mask2 = torch.where(X > -2, 0.7, 0.0)
    return mask1*phi + mask2*gamma


def sp(X, lam):
    a = []
    for el in X:
        if el > 0:
            a.append((el.pow(lam)-1)/lam)
        elif el < 0:
            a.append((-1*(-el).pow(lam) - 1) / lam)
        else:
            a.append(el.log())
    return torch.stack(a)


import torch
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
X = torch.tensor(np.arange(-10, 10, 0.01))
X_T = sp(X, 0.5)

import matplotlib.pyplot as plt
original = dist(X)
plt.plot(X, original)
plt.plot(X_T, dist(X_T))

