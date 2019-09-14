#!/usr/bin/env python3

"""Simulate vehicle motion.

Author: Andrew Adare
Jan 2018
"""

from time import time, sleep

import numpy as np

from ssd_robotics import Vehicle, draw


def main():
    dt = 0.05  # time step - 20 Hz updates
    n_steps = 2000
    extents = (-100, 100)  # map boundaries (square map)
    n_landmarks = 5
    speed = 10.0  # commanded speed in m/s (1 m/s = 2.237 mph)

    # Start in the middle, pointed in a random direction.
    starting_pose = np.array([0, 0, np.random.uniform(0, 2 * np.pi)])
    vehicle = Vehicle(
        wheelbase=0.7,
        center_of_mass=0.35,
        initial_state=starting_pose,
        range_noise=0.1,  # meters
        bearing_noise=0.01,  # rad
        sensor_range=50,
    )
    vehicle.t = time()

    # Add landmarks to the map at random.
    landmarks = np.random.uniform(*extents, size=(n_landmarks, 2))

    # ukf = vehicle_ukf(vehicle, landmarks, dt)

    for i in range(n_steps):
        start = vehicle.t
        steer_angle = np.radians(1.0 * np.sin(i / 100))
        u = np.array([speed, steer_angle])
        u_noise = np.array([0.1, 0.01])
        vehicle.move(u=u, u_noise=u_noise, dt=dt, extents=extents)
        # ukf.predict(fx_args=(u, extents))
        z, in_range = vehicle.observe(landmarks)
        # ukf.update(z, hx_args=(landmarks[in_range],))
        vehicle.t = time()
        if start + dt > vehicle.t:
            sleep(start + dt - vehicle.t)

        draw(
            vehicle.x,
            landmarks=landmarks,
            observations=z,
            x_extents=extents,
            y_extents=extents,
        )


if __name__ == "__main__":
    main()
