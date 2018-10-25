import numpy as np
import matplotlib.pyplot as plt


def add_arrow(line, position=None, direction='right', size=10, color=None, zorder=2):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color, zorder=zorder),
        size=size
    )

def make_point_arrow(point, z):
    a = np.linspace(point, point + -g[int(100 * point)] * 0.04, int(abs(g[int(100 * point)]) * 200))
    print(g[int(100 * point)])
    b = a * g[int(100 * point)]
    b += -b[0] + y[int(100 * point)] + 0.05
    line, = ax.plot(a, b, visible=False)
    print(line.get_color())
    ax.plot(a[0], b[0], 'bo', markersize=7, c=line.get_color(), zorder=z)
    add_arrow(line, position=a[-2], color=line.get_color(), zorder=z)
    line2, = ax.plot(a[:-10], b[:-10], c=line.get_color(), zorder=z, linewidth=1)

    return a[-1]

x = np.linspace(0, 5, 500)

fig, ax = plt.subplots()

y = -0.841667 *x**5 + 10.9167 *x**4 - 50.2917 *x**3 + 97.0833 *x**2 - 65.8667 *x + 10.

g = -4.20834 *x**4 + 43.6668 *x**3 - 150.875 *x**2 + 194.167 *x - 65.8667

line3, = ax.plot(x, y, zorder=-1000)
start = 2.1
for z in range(10, 100, 10):
    start = make_point_arrow(start, -z)
ax.set_xlim(1.7, 3.7)
ax.set_ylim(7, 12.5)
ax.set_xlabel("x")
ax.set_ylabel("C(x)")
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
fig.savefig('gradient_full.pdf', bbox_inches='tight', transparent=True)
# plt.show()
