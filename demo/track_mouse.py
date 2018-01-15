#!/usr/bin/env python3

"""Interactive mouse pointer tracking using a basic linear Kalman filter.
Author: Andrew Adare
"""

import sys
from collections import deque

import cv2
import numpy as np
from scipy.linalg import block_diag
from numpy.linalg import norm, eig
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def point_2d_kalman_filter(initial_state, Q_std, R_std):
    """Factory function returning a linear Kalman filter instance whose state
    vector tracks position and velocity of a 2D point. Only (x, y) are directly
    observed; velocity is latent in the motion model.

    Parameters
    ----------
    initial_state : sequence of floats
        [x0, vx0, y0, vy0]
    Q_std : float
        Standard deviation to use for process noise covariance matrix
    R_std : float
        Standard deviation to use for measurement noise covariance matrix

    Returns
    -------
    kf : filterpy.kalman.KalmanFilter instance
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0   # time step

    # State mean (x, vx, y, vy) and covariance
    kf.x = np.array([initial_state]).T
    kf.P = np.eye(kf.dim_x) * 500.

    # No control inputs
    kf.u = 0.

    # State transition matrix
    kf.F = np.array([[1, dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dt],
                     [0, 0, 0, 1]])

    # Measurement matrix - maps from state space to observation space
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])

    # Measurement noise covariance
    kf.R = np.eye(kf.dim_z) * R_std**2

    # Process noise covariance
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    kf.Q = block_diag(q, q)

    return kf


class MouseTracker():
    def __init__(self, im, sigma_proc=0.1, sigma_meas=10,
                 slider_range=(0, 50), window_name='tracking',
                 num_tail_points=50):
        self.im = im
        self.sigma_proc = sigma_proc
        self.sigma_meas = sigma_meas
        self.max_sigma_proc = 2*sigma_proc
        self.max_sigma_meas = 2*sigma_meas
        self.is_tracking = False
        self.kf = None  # Kalman filter initialized by a left click
        self.num_tail_points = num_tail_points
        self.obs_points = deque()
        self.kf_points = deque()
        self.mouse_callbacks = {
            cv2.EVENT_LBUTTONDOWN: self.on_click,
            cv2.EVENT_RBUTTONDOWN: self.on_right_click,
            cv2.EVENT_MOUSEMOVE: self.on_mousemove,
        }
        self.slider_callbacks = {
            'process noise': self.on_sigma_proc_change,
            'sensor noise': self.on_sigma_meas_change,
        }
        self.slider_range = slider_range  # in arbitrary "tick" units.
        self.window_name = window_name

        cv2.setMouseCallback(window_name, self.on_mouse)

        lo, hi = self.slider_range
        for k, v in self.slider_callbacks.items():
            cv2.createTrackbar(k, window_name, (hi - lo)//2, hi, v)

    @property
    def kx(self):
        """Kalman x-position. Read-only."""
        return self.kf.x[0, 0]

    @property
    def kvx(self):
        """Kalman x-velocity. Read-only."""
        return self.kf.x[1, 0]

    @property
    def ky(self):
        """Kalman y-position. Read-only."""
        return self.kf.x[2, 0]

    @property
    def kvy(self):
        """Kalman y-velocity. Read-only."""
        return self.kf.x[3, 0]

    def reset_kf(self, x, y):
        self.kf = point_2d_kalman_filter([x, 0., y, 0.],
                                         self.sigma_proc, self.sigma_meas)

    def on_click(self, *args):
        """Toggle tracking state."""
        x, y = args[0], args[1]
        if self.is_tracking:  # clear history and stop tracking
            self.obs_points = deque()
            self.kf_points = deque()
        else:
            self.reset_kf(x, y)
        self.is_tracking = not self.is_tracking

    def on_right_click(self, *args):
        """Reset trackbars to default position."""
        lo, hi = self.slider_range
        mid = (hi - lo)//2
        cv2.setTrackbarPos('process noise', self.window_name, mid)
        cv2.setTrackbarPos('sensor noise', self.window_name, mid)
        self.sigma_proc = self.map_linear(mid, hi, lo, 1e-6,
                                          self.max_sigma_proc)
        self.sigma_meas = self.map_linear(mid, hi, lo, 1e-6,
                                          self.max_sigma_meas)

    def on_mousemove(self, *args):
        """Perform a Kalman iteration when mouse pointer moves."""
        x, y = args[0], args[1]
        while len(self.obs_points) > self.num_tail_points:
            self.obs_points.popleft()

        if self.is_tracking:

            self.kf.predict()
            self.kf.update(np.array([[x], [y]]))

            self.obs_points.append((x, y))
            self.kf_points.append((self.kx, self.ky))

            if len(self.kf_points) > self.num_tail_points:
                self.kf_points.popleft()

    def on_mouse(self, event, x, y, flags, params):
        """Dispatch to the callback assigned to this mouse event, if any."""
        handler = self.mouse_callbacks.get(event, None)
        if handler is not None:
            handler(x, y)

    def on_sigma_proc_change(self, val):
        """Slider callback for process noise sigma parameter"""
        std = self.map_linear(val, *self.slider_range, 1e-6,
                              self.max_sigma_proc)
        self.sigma_proc = std
        if self.kf is not None:
            q = Q_discrete_white_noise(dim=2, dt=1.0, var=std**2)
            self.kf.Q = block_diag(q, q)

    def on_sigma_meas_change(self, val):
        """Slider callback for measurement noise sigma parameter"""
        std = self.map_linear(val, *self.slider_range, 1e-6,
                              self.max_sigma_meas)
        self.sigma_meas = std
        if self.kf is not None:
            self.kf.R = np.eye(self.kf.dim_z) * std**2

    def map_linear(self, x, x1, x2, y1, y2):
        """Returns y(x) evaluated from a point-slope linear function defined by
        the two intervals [x1, x2] and [y1, y2]. x need not lie in [x1, x2].
        """
        m = (y2 - y1)/(x2 - x1)
        return y1 + m*(x - x1)

    def covariance_ellipse(self, pair='x-y'):
        """Return center position, 1-sigma axes, and orientation of uncertainty
        ellipse from the x and y components of the state covariance matrix

        Parameters
        ----------
        None

        Returns
        -------
        x, y : float
            center of ellipse
        a, b : float
            semimajor and semiminor axis lengths (1 sigma)
        phi : float
            angle of semimajor axis w.r.t. the x axis in radians
        """
        if pair == 'x-y':
            P = self.kf.P[0:4:2, 0:4:2]  # x-y covariance matrix
        elif pair == 'x-vx':
            P = self.kf.P[0:2, 0:2]
        elif pair == 'y-vy':
            P = self.kf.P[2:4, 2:4]
        else:
            raise ValueError('Invalid pair keyword arg: {}'.format(pair))

        lambdas, vs = eig(P)
        a, b = np.sqrt(lambdas)  # semimajor and semiminor axes

        # Ellipse rotation angle
        # I chose -y because y is down in image coords.
        phi = np.arctan2(-vs[1], vs[0])[0]

        x, y = self.kx, self.ky  # center point
        return x, y, a, b, phi

    def draw_track(self, track, color):
        if len(track) == 0:
            return
        x, y = track[-1]
        cv2.circle(self.im, (int(x), int(y)), 4, color, -1, 4)
        cv2.polylines(self.im, np.int0([track]), False, color, thickness=2)

    def draw_ellipse(self, x, y, a, b, phi,
                     color=(255, 255, 255), nsigma=3.0, scale=(1.0, 1.0)):
        center = [int(s*x) for (s, x) in zip(scale, (x, y))]
        axes = [int(np.abs(s)*x)
                for (s, x) in zip(scale, (nsigma*a, nsigma*b))]
        angle = int(phi*180/np.pi)
        cv2.ellipse(self.im, tuple(center), tuple(axes), angle, 0, 360,
                    color, 2)

    def draw(self):
        h, w, channels = self.im.shape
        self.im *= 0  # clear the canvas
        red = (0, 0, 255)
        magenta = (255, 0, 255)
        green = (0, 255, 0)
        yellow = (0, 240, 240)
        blue = (255, 150, 0)
        white = 3*(255,)
        self.draw_track(self.obs_points, color=blue)
        self.draw_track(self.kf_points, color=red)
        status = 'tracking' if self.is_tracking else 'click to start tracking'
        cv2.putText(self.im, 'process noise: {:.2g}'.format(self.sigma_proc),
                    (w - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)
        cv2.putText(self.im, 'sensor noise: {:.2g}'.format(self.sigma_meas),
                    (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)
        cv2.putText(self.im, status,
                    (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)

        if self.kf is None or not self.is_tracking:
            return

        # Draw covariance ellipse around tracked position
        scale = (1., h/w)  # correct for aspect ratio distortion if h != w
        nsigma = 3.0
        self.draw_ellipse(*self.covariance_ellipse(pair='x-y'), color=yellow)

        # Show position vs velocity covariance: x vs. vx and y vs. vy
        cv2.putText(self.im, 'x vs. vx:',
                    (w - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, magenta, 2)
        _, _, a, b, phi = self.covariance_ellipse(pair='x-vx')
        self.draw_ellipse(w - 180, 150, a, b, phi, color=magenta)

        cv2.putText(self.im, 'y vs. vy:',
                    (w - 100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
        _, _, a, b, phi = self.covariance_ellipse(pair='y-vy')
        self.draw_ellipse(w - 80, 150, a, b, phi, color=green)


def display(im, window_name, delay=1):
    """Display `im` in named window, blocking for `delay` ms until the next
    keypress. Note that `delay=0` blocks indefinitely."""
    cv2.imshow(window_name, im)
    char = cv2.waitKey(delay)
    if char == ord(' '):
        cv2.imwrite('snapshot.png', im)
    if char == ord('q'):
        cv2.destroyAllWindows()
        sys.exit(0)


def main():
    window_name = 'tracking'
    cv2.namedWindow(window_name)
    w, h = 800, 800
    im = np.zeros([h, w, 3])  # canvas image

    mouse_tracker = MouseTracker(im)

    print('Left click: start/stop tracking.')
    print('Right click: reset trackbars.')
    print('Press q to quit.')

    while True:
        mouse_tracker.draw()
        display(im, window_name, delay=1)


if __name__ == '__main__':
    main()
