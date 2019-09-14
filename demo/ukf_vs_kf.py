"""Unscented Kalman filter example from "Kalman and Bayesian Filters in Python"
by R. Labbe. This minimal demo shows agreement between the standard KF and the
UKF
"""

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter


def f(x, dt):
    return np.array([x[0] + x[1] * dt, x[1], x[2] + x[3] * dt, x[3]])


def h(x):
    return np.array([x[0], x[2]])


def run_standard_kf(zs, dt=1.0, std_x=0.3, std_y=0.3):
    kf = KalmanFilter(4, 2)
    kf.x = np.array([0.0, 0.0, 0.0, 0.0])
    kf.R = np.diag([std_x ** 2, std_y ** 2])
    kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

    kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
    kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)

    xs, *_ = kf.batch_filter(zs)

    return xs


def run_ukf(zs, dt=1.0):
    sigmas = MerweScaledSigmaPoints(4, alpha=0.1, beta=2.0, kappa=1.0)
    ukf = UKF(dim_x=4, dim_z=2, fx=f, hx=h, dt=dt, points=sigmas)
    ukf.x = np.array([0.0, 0.0, 0.0, 0.0])
    ukf.R = np.diag([0.09, 0.09])
    ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
    ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)

    uxs = []
    for z in zs:
        ukf.predict()
        ukf.update(z)
        uxs.append(ukf.x.copy())
    uxs = np.array(uxs)

    return uxs


def main():
    np.random.seed(1234)
    std_x, std_y = 0.3, 0.3
    zs = [np.array([i + randn() * std_x, i + randn() * std_y]) for i in range(100)]
    xs = run_standard_kf(zs)
    uxs = run_ukf(zs)

    print("UKF standard deviation {:.3f} meters".format(np.std(uxs - xs)))
    plt.figure()
    plt.plot(xs[:, 0], xs[:, 2], lw=4, label="Vanilla KF")
    plt.plot(uxs[:, 0], uxs[:, 2], lw=4, ls="--", label="UKF")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simple linear tracking problem - KF and UKF")
    plt.legend()
    plt.show()


main()
