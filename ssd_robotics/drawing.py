
import numpy as np
import gr  # Fast, simple plotting library

from ssd_robotics import covariance_ellipse


def init_plot_window(xmin, xmax, ymin, ymax):
    gr.clearws()
    gr.setwsviewport(0.0, 0.25, 0.0, 0.25)  # Desktop window extents in meters
    gr.setviewport(0.15, 0.95, 0.15, 0.95)
    gr.setwindow(xmin, xmax, ymin, ymax)


def draw_vehicle(x):
    """Draw the vehicle CM as a circle. Draw a line for the vehicle heading.
    x = [x, y, phi]
    """
    gr.setmarkertype(gr.MARKERTYPE_CIRCLE)
    gr.setmarkersize(2)
    gr.setmarkercolorind(1)  # black
    gr.polymarker([x[0]], [x[1]])
    xh = [x[0], x[0] + 5*np.cos(x[2])]
    yh = [x[1], x[1] + 5*np.sin(x[2])]
    gr.polyline(xh, yh)


def draw_vehicle_ellipse(x, P):
    ell = covariance_ellipse(x[0:2], P[0:2, 0:2], n_sigma=2)
    gr.settransparency(0.8)
    gr.setlinewidth(2)
    # gr.setlinecolorind(983)
    gr.polyline(ell[0, :], ell[1, :])
    gr.setlinewidth(1)
    gr.settransparency(1.0)


def draw_particles(xs, weights=None):
    if weights is not None:
        indices = np.argsort(np.array(weights))
        alphas = indices/len(weights)
    else:
        alphas = np.full((len(xs),), 0.5)
    gr.setmarkertype(gr.MARKERTYPE_SOLID_CIRCLE)
    gr.setmarkercolorind(2)  # red
    gr.setmarkersize(0.75)
    for x, t in zip(xs, alphas):
        gr.settransparency(t)
        gr.polymarker([x[0]], [x[1]])
    gr.settransparency(1.0)


def draw_landmarks(landmarks):
    gr.setmarkertype(gr.MARKERTYPE_SOLID_CIRCLE)
    gr.setmarkersize(2)
    gr.setmarkercolorind(4)  # blue
    gr.polymarker(landmarks[:, 0], landmarks[:, 1])


def draw_observation_lines(x, observations):
    for (r, b) in observations:
        x1, x2 = x[0], x[0] + r*np.cos(b + x[2])
        y1, y2 = x[1], x[1] + r*np.sin(b + x[2])
        gr.polyline([x1, x2], [y1, y2])


def draw_axes(tick_spacing, xmin, ymin):
    gr.axes(tick_spacing, tick_spacing, xmin, ymin, 5, 5, -0.01)
    midway = 0.54
    gr.textext(midway, 0.02, 'x')
    gr.setcharup(-1, 0)  # Vertical, end-up
    gr.textext(0.05, midway, 'y')
    gr.setcharup(0, 1)  # Back to horizontal


def draw(x,
         x_extents=(-100, 100),
         y_extents=(-100, 100),
         landmarks=None,
         observations=None,
         particles=None,
         weights=None,
         fig=None):
    """Draw vehicle state x = [x, y, theta] on the map."""
    xmin, xmax = x_extents
    ymin, ymax = y_extents
    tick_spacing = (xmax - xmin)/20

    if fig is not None:
        print('saving', draw.i)
        gr.beginprint('{}_{}.pdf'.format(fig, draw.i))

    init_plot_window(xmin, xmax, ymin, ymax)
    draw_vehicle(x)
    # draw_vehicle_ellipse(x, np.eye(3))  # this works; not needed yet

    if landmarks is not None:
        draw_landmarks(landmarks)

    if observations is not None:
        draw_observation_lines(x, observations)

    if particles is not None:
        draw_particles(particles, weights=weights)

    draw_axes(tick_spacing, xmin, ymin)

    gr.updatews()
    if fig is not None:
        gr.endprint()
    draw.i += 1
    return


draw.i = 0
