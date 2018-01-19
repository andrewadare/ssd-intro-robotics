import numpy as np
from scipy.linalg import sqrtm


def mean_angle(angles):
    """Compute the mean of `angles`, accounting for periodicity."""
    return np.angle(np.sum(np.exp(1j*np.asarray(angles))))


def mpi_to_pi(phi):
    """Ensure that phi resides within [-pi, pi]"""
    phi %= 2*np.pi
    if isinstance(phi, np.ndarray):
        phi[phi > +np.pi] -= 2*np.pi
        phi[phi < -np.pi] += 2*np.pi
    elif phi > np.pi:
        phi -= 2 * np.pi
    return phi


def in2pi(a):
    """Ensure that a resides within [0, 2pi]"""
    return (a + 2*np.pi) % (2*np.pi)


def rk4(t, dt, x, deriv):
    """Fourth-order Runge-Kutta integrator.

    Parameters
    ----------
    t : float
        time
    dt : float
        time step
    x : float or broadcastable sequence
        state vector
    deriv : function(x, t)
        callback to compute xdot at time t

    Returns
    -------
    (x_new, t_new) : new state and time
    """
    k1 = dt*deriv(x, t)
    k2 = dt*deriv(x + k1/2, t + dt/2)
    k3 = dt*deriv(x + k2/2, t + dt/2)
    k4 = dt*deriv(x + k3, t + dt)
    return x + (k1 + 2*(k2 + k3) + k4)/6, t + dt


def wrap(x, extents):
    """Wrap first two components of x into extents."""
    a, b = extents
    d = b - a
    if x[0] < a:
        x[0] += d
    if x[1] < a:
        x[1] += d
    if x[0] > b:
        x[0] -= d
    if x[1] > b:
        x[1] -= d
    return x


def covariance_ellipse(x, P, nsigma=3, nsegs=16):
    """Compute 2D ellipse centered at (x[0], x[1]) using covariance matrix P.
    Approximated with nsegs line segments (nsegs + 1 points)."""
    phi = np.linspace(0, 2*np.pi, nsegs + 1)
    ex = np.cos(phi)[np.newaxis, :]
    ey = np.sin(phi)[np.newaxis, :]
    return np.dot(nsigma*sqrtm(P), np.vstack([ex, ey])) + x[:2].reshape((2, 1))
