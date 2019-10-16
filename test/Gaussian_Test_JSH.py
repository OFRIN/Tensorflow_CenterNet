
import math
import numpy as np

def gaussian2D(shape, p, sigma_x, sigma_y):
    w, h = shape
    px, py = p[0], p[1]

    x = np.arange(w) + 0.5
    y = np.arange(h) + 0.5
    x, y = np.meshgrid(x, y)

    y = np.exp(-((x - px)**2 + (y - py)**2) / (2 * sigma_x * sigma_y))
    
    # exception
    # y[y < 0] = 0
    # y[y > 1] = 1

    return y

def generate_gaussian(points, radius_x, radius_y):
    gaussian = gaussian2D((radius_x, radius_y), points, sigma_x = radius_x / 6., sigma_y = radius_y / 6.)
    return gaussian

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # case 1 - JSH
    fig = plt.figure()
    ax = fig.gca(projection = "3d")

    gaussian = generate_gaussian([100, 150], 200, 300)

    h, w = gaussian.shape

    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, y, gaussian, cmap = cm.coolwarm, linewidth = 0, antialiased = False)

    plt.show()
