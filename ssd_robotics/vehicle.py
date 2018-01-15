"""Vehicle class to model wheeled motion in two dimensions.

Author: Andrew Adare
Jan 2018
"""

import numpy as np

from ssd_robotics import rk4, mpi_to_pi, wrap


class Vehicle():
    def __init__(self,
                 wheelbase=0.7,
                 center_of_mass=0.35,
                 initial_state=np.zeros(3),  # [x, y, theta]
                 sensor_range=np.inf,
                 range_noise=0.,
                 bearing_noise=0.,
                 ):
        """
        Parameters
        ----------
        wheelbase : float
            Distance between front and rear axles in meters
        center_of_mass : float
            Position of CM along centerline in meters
        initial_state : ndarray - shape (3,)
            initial [x, y, theta] state vector
        sensor_range : float (optional, default infinity)
            Maximum detection distance
        range_noise, bearing_noise : floats (optional, default 0)
            Gaussian sigma parameters representing range and angle resolution
        """
        self.x = initial_state
        self.u = [0., 0.]  # control inputs: speed and steering angle
        self.a = center_of_mass
        self.b = wheelbase
        self.t = 0.  # on-board time marker
        self.range_sigma = range_noise
        self.bearing_sigma = bearing_noise
        self.sensor_range = sensor_range

    def eqs_of_motion(self, state, t):
        """Kinematic bicycle model.
        Reference: "Feedback systems" by K. Astrom & R. Murray or
        "Vehicle Dynamics and Control" by R. Rajamani.
        """
        x, y, theta = state
        v0, delta = self.u  # throttle and steering commands
        alpha = np.arctan2(self.a*np.tan(delta), self.b)
        xdot = v0*np.cos(alpha + theta)/np.cos(alpha)
        ydot = v0*np.sin(alpha + theta)/np.cos(alpha)
        tdot = v0/self.b * np.tan(delta)
        return np.array([xdot, ydot, tdot])

    def move(self, u=np.zeros(2), u_noise=np.zeros(2), dt=0.1, extents=None):
        """Simulate vehicle motion from control inputs and kinematic
        equations.

        Parameters
        ----------
        u : ndarray - shape (2,)
            control input vector [speed (m/s), steering angle (rad)]
        u_noise : ndarray - shape (2,)
            Sigma parameters characterizing uncertainy on u
        dt : float (optional, default 0.1)
            Time step size (sec)
        extents : tuple(float, float) (optional, default None)
            Boundaries of the square, periodic robot world.
        """
        self.u = u + np.random.randn()*u_noise
        self.x, self.t = rk4(self.t, dt, self.x, self.eqs_of_motion)
        if extents is not None:
            self.x = wrap(self.x, extents)

    def observe(self, landmarks):
        """Observe ranges and bearings to landmarks from current pose.

        Parameters
        ----------
        landmarks : ndarray - shape (n_landmarks, 2)
            (x, y) landmark positions

        Returns
        -------
        z : ndarray - shape (n_landmarks, 2)
            Observation vector with ranges in first column and bearings in
            second column.
        """
        n = landmarks.shape[0]
        d = landmarks - self.x[np.newaxis, :2]  # vehicle-to-landmark dx, dy

        ranges = np.linalg.norm(d, axis=1) + \
            self.range_sigma*np.random.randn(n)
        bearings = np.arctan2(d[:, 1], d[:, 0]) - self.x[2] + \
            self.bearing_sigma*np.random.randn(n)

        in_range = ranges < self.sensor_range
        ranges, bearings = ranges[in_range], mpi_to_pi(bearings[in_range])

        z = np.vstack([ranges, bearings]).T

        return z, in_range
