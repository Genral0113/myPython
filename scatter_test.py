import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt


def parametric_curve():
    mpl.rcParams['legend.fontsize'] = 10

    ax = plt.axes(projection='3d')

    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)

    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    ax.plot(x, y, z, label='parametric curve')

    ax.legend()
    plt.show()


def randrange(n, vmin, vmax):
    return (vmax - vmin) * np.random.rand(n) + vmin


def scatter_plots():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100

    for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = randrange(n, 23, 23)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, c=c, marker=m)

    plt.show()


def wireframe_plots():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y, Z = axes3d.get_test_data(0.05)

    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    plt.show()


def surface_plots():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(-1.01, 2.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def tri_surface_plots():
    n_radii = 8
    n_angles = 36

    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

    x = np.append(0, (radii * np.cos(angles)).flatten())
    y = np.append(0, (radii * np.sin(angles)).flatten())

    z = np.sin(-x * y)

    ax = plt.axes(projection='3d')

    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

    plt.show()


def contour_plots():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y, Z = axes3d.get_test_data(0.05)
    cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
    ax.clabel(cset, fontsize=9, inline=1)

    plt.show()


def contour_plots1():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)

    cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_xlim(-40, 40)
    ax.set_ylabel('Y')
    ax.set_ylim(-40, 40)
    ax.set_zlabel('Z')
    ax.set_zlim(-100, 100)

    plt.show()


def contour_plots2():

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)

    cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_xlim(-40, 40)
    ax.set_ylabel('Y')
    ax.set_ylim(-40, 40)
    ax.set_zlabel('Z')
    ax.set_zlim(-100, 100)

    plt.show()


def bar_plots():

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.random.rand(20)

        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def subplot():

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = np.linspace(0, 1, 100)
    y = np.sin(x * 2 * np.pi) / 2 + 0.5

    ax.plot(x, y, zs=0, zdir='z', label='curve in (x,y)')

    colors = ('r', 'g', 'b', 'k')
    x = np.random.sample(20 * len(colors))
    y = np.random.sample(20 * len(colors))
    c_list = []
    for c in colors:
        for i in range(20):
            c_list.append(c)

    ax.scatter(x, y, zs=0, zdir='y', c=c_list, label='points in (x,z)')

    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def subplot1():
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    plt.show()


def plot_text():

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    xs = (1, 4, 4, 9, 4, 1)
    ys = (2, 5, 8, 10, 1, 2)
    zs = (10, 3, 8, 9, 1, 8)

    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
        ax.text(x, y, z, label, zdir)

    ax.text(9, 0, 0, "red", color='red')

    ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()


if __name__ == '__main__':
    parametric_curve()

    scatter_plots()

    wireframe_plots()

    surface_plots()

    tri_surface_plots()

    contour_plots()
    contour_plots1()
    contour_plots2()

    bar_plots()
    subplot1()
    plot_text()
