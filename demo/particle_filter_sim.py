#!/usr/bin/env python3

"""Particle filter localization example.

Author: Andrew Adare
Jan 2018
"""

from time import time, sleep
from collections import deque

import numpy as np
import scipy
from scipy.stats import multivariate_normal as mvn

from ssd_robotics import \
    Vehicle, draw, mpi_to_pi, in2pi, sample_x_using_odometry


def update(xs, w, measurements, landmarks, R):
    assert len(measurements) == len(landmarks)
    w.fill(1.0)

    for lm, z in zip(landmarks, measurements):

        # dx, dy array - shape (N, 2)
        deltas = lm - xs[:, :2]

        # Range and bearing measurements - shapes (N,)
        r = np.linalg.norm(deltas, axis=1)
        b = mpi_to_pi(np.arctan2(deltas[:, 1], deltas[:, 0]) - xs[:, 2])

        # This broadcasts evaluation over N gaussians, each with a different
        # mean at r[j], at the value z[0].
        w *= scipy.stats.norm(r, R[0, 0]).pdf(z[0])
        # same thing but for bearing measurements
        w *= scipy.stats.norm(b, R[1, 1]).pdf(z[1])

        # This is a failed attempt to sample from a multivariate normal using
        # broadcasting as above.
        # mu = np.vstack([r, b])
        # w *= mvn(mu, R).pdf(z)

    w += 1.e-300  # avoid round-off to zero
    w /= sum(w)  # normalize
    return


def likelihood(x, measurements, landmarks, R):
    """Currently unused"""
    assert len(measurements) == len(landmarks)
    prob = 1.0
    for lm, z in zip(landmarks, measurements):
        deltas = lm - x[:2]  # dx, dy
        r = np.linalg.norm(deltas)
        b = mpi_to_pi(np.arctan2(deltas[1], deltas[0]) - x[2])
        prob *= mvn.pdf(z, mean=np.array([r, b]), cov=R)
    return prob


def n_effective(weights):
    """Return the fraction of particles that have an influential weight."""
    # return np.size(weights)/np.sum(np.square(weights))
    return 1/(np.sum(weights**2) + 1e-300)


def resample(particles, weights):
    resampled = []
    N = len(particles)
    index = np.random.randint(N)
    beta = 0
    betamax = 2*max(weights)

    for _ in range(N):
        beta += np.random.uniform(0, betamax)
        while weights[index] < beta:
            beta -= weights[index]
            index = (index + 1) % N
        resampled.append(particles[index].copy())

    return resampled


def main():
    dt = 0.05  # time step - 20 Hz updates
    n_steps = 1000
    extents = (-100, 100)  # map boundaries (square map)
    L = extents[1] - extents[0]
    n_landmarks = 5
    speed = 20.  # commanded speed in m/s (1 m/s = 2.237 mph)
    N = 500  # number of particles
    u_noise = np.array([0.1*speed, 0.001])  # speed noise, steering noise

    # Start in the middle, pointed in a random direction.
    starting_pose = np.array([0, 0, np.random.uniform(0, 2*np.pi)])
    vehicle = Vehicle(wheelbase=0.7,
                      center_of_mass=0.35,
                      initial_state=starting_pose,
                      range_noise=0.1,  # meters
                      bearing_noise=0.01,  # rad
                      sensor_range=50)
    vehicle.t = time()

    particles = []
    weights = np.full(shape=(N,), fill_value=1/N)
    R = np.diag([vehicle.range_sigma, vehicle.bearing_sigma])

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

        # Initialize particle distribution
        x0 = np.array([0.1*L, 0.1*L, 0.01])*np.random.randn(3) + starting_pose
        particles.append(x0)

    # Add landmarks to the map at random.
    landmarks = np.random.uniform(*extents, size=(n_landmarks, 2))

    for i in range(n_steps):
        start = vehicle.t
        steer_angle = np.radians(1.0*np.sin(i/100))
        u = np.array([speed, steer_angle])

        # Advance the ground-truth pose
        x_prev = vehicle.x.copy()  # Previous global state
        vehicle.move(u=u, u_noise=u_noise, dt=dt, extents=extents)

        z, in_range = vehicle.observe(landmarks)

        # predict
        for j in range(N):
            u_odom[j].popleft()

            # Simulate a state \bar{x}_t measured by odometry.
            xbar = vehicle.x - x_prev + u_odom[j][-1]
            xbar[2] = in2pi(xbar[2])
            u_odom[j].append(xbar)

            # Update particle position with a new prediction.
            particles[j] = sample_x_using_odometry(
                particles[j], u_odom[j],
                # variances=10*np.array([1e-5, 1e-5, 1e-2, 1e-3]),
                variances=np.array([1e-4, 1e-4, 1e-1, 1e-2]),
                extents=extents)

        # update
        if len(z) > 0:
            # update(np.array([p.x for p in particles]),
            update(np.array(particles),
                   weights, z, landmarks[in_range], R)

            # resample
            if n_effective(weights) < N/2:
                particles = resample(particles, weights)

        vehicle.t = time()
        if start + dt > vehicle.t:
            sleep(start + dt - vehicle.t)

        # pdf = 'pf-sim' if i < 10 else None
        pdf = None
        draw(vehicle.x,
             landmarks=landmarks,
             observations=z,
             x_extents=extents,
             y_extents=extents,
             particles=particles,
             weights=weights,
             fig=pdf
             )


if __name__ == '__main__':
    main()
