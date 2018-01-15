import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()


def plot_hist(x1, x2, fig, ax, i):
    ax.clear()
    plt.hist(x1, bins=100, histtype='step', normed=True)
    plt.hist(x2, bins=100, histtype='step', normed=True)
    plt.title('time step {}'.format(i))
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.show(block=False)
    input()  # press enter to move on


def f_lin(x):
    return x+1


def f_nonlin(x):
    return + x + 1 + 0.2*np.cos(3*x)


def plot_f():
    plt.figure()
    x = np.arange(-5, 5, 0.01)
    plt.plot(x, f_lin(x))
    plt.plot(x, f_nonlin(x))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('State update functions')
    plt.savefig('lin_nonlin.pdf')


def main():
    xlin = np.random.randn(int(1e5))
    xnon = np.random.randn(int(1e5))
    print('press enter to continue.')
    for i in range(10):
        plot_hist(xlin, xnon, fig, ax, i)
        xlin = f_lin(xlin)
        xnon = f_nonlin(xnon)
        plt.savefig('hist_{:02d}.pdf'.format(i))


# plot_f()
main()
