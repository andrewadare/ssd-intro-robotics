#!/usr/bin/env python3

"""Predict next vehicle state by sampling from a state transition probability
density based on a general odometry model.

Author: Andrew Adare
Jan 2018
"""

from time import time, sleep
from collections import deque

import numpy as np
from numpy.random import randn

from ssd_robotics import Vehicle, draw, wrap


def in2pi(a):
    return (a + 2*np.pi) % (2*np.pi)


def sample_state(x, u, variances=[0.01, 0.01, 0.01, 0.01], extents=None):
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


def main():
    dt = 1.0  # time step
    n_steps = 2000
    extents = (-100, 100)  # map boundaries (square map)
    speed = 20.  # commanded speed in m/s (1 m/s = 2.237 mph)

    # I tried these combinations of variances for odometry sampling:
    # v = [0.1, 0.0, 0.0, 0.0]  # arc
    # v = [0.0, 0.0001, 0.0, 0.0]  # arc
    # v = [0.1, 0.0001, 0.0, 0.0]  # arc
    # v = [0.0, 0.0, 0.01, 0.0]  # straight-ahead line
    # v = [0.1, 0.0, 0.01, 0.0]  # teardrop, ring, line (angle-dependent)
    # v = [0.001, 0.0001, 0.001, 0.0] # banana
    # v = [0.001, 0.0001, 0.01, 0.0]  # textbook
    v = [1e-3, 1e-5, 1e-3, 1e-3]  # compact blob

    # Start in the middle, pointed in a random direction.
    starting_pose = np.array([0, 0, np.random.uniform(0, 2*np.pi)])
    vehicle = Vehicle(wheelbase=0.7,
                      center_of_mass=0.35,
                      initial_state=starting_pose,
                      range_noise=0.1,  # meters
                      bearing_noise=0.01,  # rad
                      sensor_range=50)
    vehicle.t = time()

    # Use control input vector to store odometry info. There will be two
    # entries in u: x_{t-1} and x_t. Each is a state in the local frame
    # of the vehicle (x_bar, y_bar, theta_bar). The starting point is
    # arbitrary; we can only measure changes.
    u = deque([np.array([np.random.uniform(*extents),
                         np.random.uniform(*extents),
                         np.random.uniform(0, 2*np.pi)
                         ])])

    vehicle.move(u=np.array([speed, 0]), dt=dt, extents=extents)
    u.append(vehicle.x - starting_pose + u[-1])
    u[-1][2] = in2pi(u[-1][2])

    for i in range(n_steps):
        steer_angle = np.radians(1.0*np.sin(0.5*i/np.pi))

        x_prev = vehicle.x.copy()  # Previous global state
        vehicle.move(u=np.array([speed, steer_angle]), dt=dt, extents=extents)

        # Simulate a state \bar{x}_t measured by odometry.
        u.append(vehicle.x - x_prev + u[-1])
        u[-1][2] = in2pi(u[-1][2])
        u.popleft()

        x_t = [sample_state(x_prev, u, variances=v, extents=extents)
               for _ in range(500)]

        sleep(dt)

        draw(x_prev,
             x_extents=extents,
             y_extents=extents,
             particles=x_t,
             )


if __name__ == '__main__':
    main()
