#!/usr/bin/env python3

"""This is a simulation of the prediction phase in a particle filter. The
particles quickly disperse because there is no update from observations to
constrain the estimates.

At each time step, the vehicle state is predicted by sampling from a state
transition probability density based on a general odometry model.

Author: Andrew Adare
Jan 2018
"""

from time import time, sleep
from collections import deque

import numpy as np

from ssd_robotics import Vehicle, draw, in2pi, sample_x_using_odometry


def main():
    dt = 0.5  # time step
    n_steps = 2000
    extents = (-100, 100)  # map boundaries (square map)
    speed = 20.  # commanded speed in m/s (1 m/s = 2.237 mph)
    N = 500
    particles = []

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

    # Use control input vector to store odometry info. u_odom is a list of N
    # queue structures. There will be two states in each entry: x_{t-1} and
    # x_t. Each is a state in the local frame of the vehicle (x_bar, y_bar,
    # theta_bar). The starting point is taken to be known; from there, we can
    # only measure changes.
    u_odom = [deque([starting_pose.copy()]) for _ in range(N)]

    # Advance the vehicle one step so we can get an odometry measurement.
    vehicle.move(u=np.array([speed, 0]), dt=dt, extents=extents)
    for j in range(N):
        xbar = vehicle.x - starting_pose + u_odom[j][-1]
        xbar[2] = in2pi(xbar[2])
        u_odom[j].append(xbar)
        particles.append(vehicle.x)

    for i in range(n_steps):
        steer_angle = np.radians(1.0*np.sin(0.5*i/np.pi))

        x_prev = vehicle.x.copy()  # Previous global state
        vehicle.move(u=np.array([speed, steer_angle]), dt=dt, extents=extents)

        for j in range(N):
            u_odom[j].popleft()

            # Simulate a state \bar{x}_t measured by odometry.
            xbar = vehicle.x - x_prev + u_odom[j][-1]
            xbar[2] = in2pi(xbar[2])
            u_odom[j].append(xbar)

            # Update particle position with a new prediction.
            particles[j] = sample_x_using_odometry(particles[j], u_odom[j],
                                                   variances=v, extents=extents)
        sleep(dt)

        draw(x_prev,
             x_extents=extents,
             y_extents=extents,
             particles=particles,
             )


if __name__ == '__main__':
    main()
