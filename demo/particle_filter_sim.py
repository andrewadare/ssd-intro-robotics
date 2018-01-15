#!/usr/bin/env python3

"""Particle filter localization example.

Author: Andrew Adare
Jan 2018
"""

from time import time, sleep

import numpy as np
import scipy
from scipy.stats import multivariate_normal as mvn
from copy import deepcopy

from ssd_robotics import Vehicle, draw, mpi_to_pi


def update(xs, w, measurements, landmarks, R):
    assert len(measurements) == len(landmarks)
    w.fill(1.0)

    for lm, z in zip(landmarks, measurements):

        # dx, dy array - shape (N, 2)
        deltas = lm - xs[:, :2]

        # Range and bearing measurements - shapes (N,)
        r = np.linalg.norm(deltas, axis=1)
        b = mpi_to_pi(np.arctan2(deltas[:, 1], deltas[:, 0]) - xs[:, 2])

        w *= scipy.stats.norm(r, R[0, 0]).pdf(z[0])
        w *= scipy.stats.norm(b, R[1, 1]).pdf(z[1])
    w += 1.e-300  # avoid round-off to zero
    w /= sum(w)  # normalize
    return


def likelihood(vehicle, measurements, landmarks, R):
    assert len(measurements) == len(landmarks)
    prob = 1.0
    for lm, z in zip(landmarks, measurements):
        deltas = lm - vehicle.x[:2]  # dx, dy
        r = np.linalg.norm(deltas)
        b = mpi_to_pi(np.arctan2(deltas[1], deltas[0]) - vehicle.x[2])
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
        resampled.append(deepcopy(particles[index]))

    return resampled


def main():
    dt = 0.05  # time step - 20 Hz updates
    n_steps = 1000
    extents = (-100, 100)  # map boundaries (square map)
    n_landmarks = 10
    speed = 10.  # commanded speed in m/s (1 m/s = 2.237 mph)
    N = 100  # number of particles
    u_noise = np.array([0.1*speed, 0.001])  # speed noise, steering noise
    t_pred, t_up, t_resample = 0., 0., 0.

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

    for _ in range(N):
        pose = 0.1*(extents[1]-extents[0])*np.random.randn(3) + starting_pose
        v = Vehicle(wheelbase=0.7, center_of_mass=0.35, initial_state=pose)
        particles.append(v)
    assert len(particles) == weights.shape[0]  # debug

    # Add landmarks to the map at random.
    landmarks = np.random.uniform(*extents, size=(n_landmarks, 2))

    for i in range(n_steps):
        start = vehicle.t
        steer_angle = np.radians(1.0*np.sin(i/100))
        u = np.array([speed, steer_angle])

        # Advance the ground-truth pose
        vehicle.move(u=u, u_noise=u_noise, dt=dt, extents=extents)

        z, in_range = vehicle.observe(landmarks)

        # predict
        start = time()
        for j, p in enumerate(particles):
            particles[j].move(u=u, u_noise=u_noise, dt=dt, extents=extents)

            # Add Gaussian noise to predicted states in addition to the control
            # noise. # Empirically, this helps to mitigate sample
            # impoverishment.
            particles[j].x += np.array([0.1, 0.1, 0.001])*np.random.randn(3)
        t_pred += time() - start

        # update
        start = time()
        if len(z) > 0:
            update(np.array([p.x for p in particles]),
                   weights, z, landmarks[in_range], R)
            # for j, p in enumerate(particles):
            #     weights[j] = likelihood(p, z, landmarks[in_range], R)
        t_up += time() - start

        # resample
        start = time()
        if n_effective(weights) < N/2:
            particles = resample(particles, weights)
        t_resample += time() - start

        vehicle.t = time()
        if start + dt > vehicle.t:
            sleep(start + dt - vehicle.t)

        # save = True if i % 100 == 0 else False
        save = False
        draw(vehicle.x,
             landmarks=landmarks,
             observations=z,
             x_extents=extents,
             y_extents=extents,
             particles=[p.x for p in particles],
             weights=weights,
             save=save
             )
    print(t_pred, t_up, t_resample)


if __name__ == '__main__':
    main()
