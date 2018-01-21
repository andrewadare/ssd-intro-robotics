#!/usr/bin/env python3

"""UKF localization using the filterpy library. It is adapted from the notebook
example from "Kalman and Bayesian Filters in Python" by R. Labbe.

The demo is not perfect. There are frequent crashes within filterpy during
Cholesky factorization, and for some reason I need to scale up R far larger
than is expected. The whole thing seems brittle overall (try adjusting
parameters to see what I mean).

Improvements and bug fixes are welcome!

Author: Andrew Adare
Jan 2018
"""

from time import time, sleep
from copy import deepcopy

import numpy as np
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise

from ssd_robotics import \
    Vehicle, draw, mpi_to_pi, wrap, mean_angle, covariance_ellipse


def create_ukf(vehicle, landmarks, P0, Q, dt, extents):

    def f(x, dt, u, extents):
        vehicle.x = x.copy()
        vehicle.move(u=u, dt=dt, extents=extents)
        return vehicle.x

    def h(x, landmark):
        hx = np.empty(2)
        lx, ly = landmark
        hx[0] = np.sqrt((lx - x[0])**2 + (ly - x[1])**2)
        hx[1] = np.arctan2(ly - x[1], lx - x[0])
        return hx

    def residual_h(z, h):
        """Subtract [range, bearing] vectors, accounting for angle"""
        y = z - h
        y[1] = mpi_to_pi(y[1])
        wrap(y, extents)
        return y

    def residual_x(xa, xb):
        """Subtract [x, y, phi] vectors, accounting for angle"""
        dx = xa - xb
        dx[2] = mpi_to_pi(dx[2])
        wrap(dx, extents)
        return dx

    def state_mean(sigmas, w_m):
        x = np.empty(3)

        # This is a hack (and not a very good one) to handle periodic boundary
        # crossings. Seems to help a little.
        std = np.std(sigmas[:, :2])
        if std > (extents[1] - extents[0])/4:
            x[:2] = sigmas[0, :2]
        else:
            x[0] = np.sum(np.dot(sigmas[:, 0], w_m))
            x[1] = np.sum(np.dot(sigmas[:, 1], w_m))

        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), w_m))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), w_m))
        x[2] = np.arctan2(sum_sin, sum_cos)

        wrap(x, extents)
        return x

    def z_mean(sigmas, w_m):
        z_count = sigmas.shape[1]
        x = np.zeros(z_count)

        for z in range(0, z_count, 2):
            sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), w_m))
            sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), w_m))

            x[z] = np.sum(np.dot(sigmas[:, z], w_m))
            x[z+1] = np.arctan2(sum_sin, sum_cos)
        return x

    points = MerweScaledSigmaPoints(n=3, alpha=0.5, beta=2, kappa=0,
                                    subtract=residual_x)
    ukf = UKF(dim_x=3,
              dim_z=2,
              fx=f,
              hx=h,
              dt=dt,
              points=points,
              x_mean_fn=state_mean,
              z_mean_fn=z_mean,
              residual_x=residual_x,
              residual_z=residual_h,
              )

    ukf.x = vehicle.x.copy()
    ukf.P = P0.copy()
    ukf.Q = Q

    # This large scale factor is theoretically unjustified.
    # TODO: without it, the sim crashes. Figure this out!
    ukf.R = 1e5*np.diag([vehicle.range_sigma**2, vehicle.bearing_sigma**2])
    # ukf.R = np.diag([vehicle.range_sigma**2, vehicle.bearing_sigma**2])

    return ukf


def main():
    dt = 0.05  # time step - 20 Hz updates
    n_steps = 2000
    extents = (-100, 100)  # map boundaries (square map)
    n_landmarks = 5
    speed = 20.  # commanded speed in m/s (1 m/s = 2.237 mph)
    ukf_step_size = 1

    # Start in the middle, pointed in a random direction.
    starting_pose = np.array([0, 0, np.random.uniform(0, 2*np.pi)])
    vehicle = Vehicle(wheelbase=0.7,
                      center_of_mass=0.35,
                      initial_state=starting_pose,
                      range_noise=0.01,  # meters
                      bearing_noise=0.01,  # rad
                      sensor_range=50)
    vehicle.t = time()

    # Add landmarks to the map at random.
    landmarks = np.random.uniform(*extents, size=(n_landmarks, 2))

    P0 = np.diag([50, 50, 0.01])
    Q = np.diag([1e-2, 1e-2, 1e-6])
    ukf = create_ukf(deepcopy(vehicle), landmarks, P0, Q, dt*ukf_step_size,
                     extents)

    for i in range(n_steps):
        start = vehicle.t
        steer_angle = np.radians(1.0*np.sin(i/100))
        u = np.array([speed, steer_angle])
        vehicle.move(u=u, dt=dt, extents=extents)

        zs, in_range = vehicle.observe(landmarks)

        if i % ukf_step_size == 0:
            ukf.predict(fx_args=(u, extents))
            for (z, lm) in zip(zs, landmarks[in_range]):
                ukf.update(z, hx_args=(lm,))

        vehicle.t = time()
        if start + dt > vehicle.t:
            sleep(start + dt - vehicle.t)

        ellipses = [covariance_ellipse(ukf.x, ukf.P, nsigma=n, nsegs=32)
                    for n in [1, 2, 3]]
        draw(vehicle.x,
             landmarks=landmarks,
             observations=zs,
             x_extents=extents,
             y_extents=extents,
             particles=ukf.sigmas_f,
             ellipses=ellipses
             )


if __name__ == '__main__':
    main()
