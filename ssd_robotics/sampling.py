import numpy as np
from numpy.random import randn

from ssd_robotics import in2pi, wrap


def sample_x_using_odometry(x, u,
                            variances=[0.01, 0.01, 0.01, 0.01],
                            extents=None):
    """Sample from p(x_t | u_t, x_{t-1}) where u is not a control vector, but
    a measurement of the current and previous states in local coordinates:
                    u = [\bar{x}_{t-1}, \bar{x}_t]^T
    """
    xt = np.empty(3)  # output vector: sampled global pose
    x_prev, y_prev, theta_prev = x  # global pose at time t-1

    du = u[1] - u[0]
    if extents is not None:
        du = wrap(du, extents)

    # dx, dy, d_theta = u[1] - u[0]
    dx, dy, d_theta = du
    a1, a2, a3, a4 = variances

    # drot1 = in2pi(np.arctan2(dy, dx) - u[0][2])
    drot1 = in2pi(np.arctan2(dy, dx) - theta_prev)
    dtran = np.sqrt(dx**2 + dy**2)
    drot2 = in2pi(d_theta - drot1)

    dr1 = drot1 - np.sqrt(a1*drot1**2 + a2*dtran**2)*randn()
    dtr = dtran - np.sqrt(a3*dtran**2 + a4*(drot1**2 + drot2**2))*randn()
    dr2 = drot2 - np.sqrt(a1*drot2**2 + a2*dtran**2)*randn()

    xt[0] = x_prev + dtr*np.cos(theta_prev + dr1)
    xt[1] = y_prev + dtr*np.sin(theta_prev + dr1)
    xt[2] = in2pi(theta_prev + dr1 + dr2)

    if extents is not None:
        xt = wrap(xt, extents)

    return xt
