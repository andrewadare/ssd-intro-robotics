"""Cartoon plot of a made-up PDF and its CDF"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as gaus


fig, ax = plt.subplots()


def plot_f():
    plt.figure()

    x = np.arange(-5, 5, 0.01)

    pdf = 0.6*gaus.pdf(x, loc=-2, scale=0.5) +\
        0.2*gaus.pdf(x, loc=1., scale=0.3) +\
        0.2*gaus.pdf(x, loc=1.8, scale=0.4)
    pdf /= 0.01*np.sum(pdf)

    plt.plot(x, pdf, label='pdf')
    plt.plot(x, 0.01*np.cumsum(pdf), label='cdf')
    plt.legend()
    plt.xlabel('x')

    print('saving inv-xform-sampling.pdf')
    plt.savefig('inv-xform-sampling.pdf')

plot_f()
