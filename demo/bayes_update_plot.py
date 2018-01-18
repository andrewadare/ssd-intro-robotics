#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as gaus


fig, ax = plt.subplots(figsize=(7, 5))


def plot_f():

    x = np.arange(-6, 6, 0.01)

    p1 = gaus.pdf(x, loc=-2, scale=1.3)
    p2 = gaus.pdf(x, loc=2.5, scale=1.0)
    p3 = p1*p2
    p3 *= 100/np.sum(p3)

    print(np.sum(p1), np.sum(p2), np.sum(p3))

    ax.fill(x, p1, label='$p(x_{t})_{pred}$', alpha=0.3)
    ax.plot(x, p1, color='black')
    ax.fill(x, p2, label='$p(z_{t} | x_{t})$', alpha=0.3)
    ax.plot(x, p2, color='black')
    ax.fill(x, p3, label='product', alpha=0.3)
    ax.plot(x, p3, color='black')

    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('pdf')

    print('saving bayes-update.pdf')
    fig.savefig('bayes-update.pdf', bbox_inches='tight')

plot_f()
