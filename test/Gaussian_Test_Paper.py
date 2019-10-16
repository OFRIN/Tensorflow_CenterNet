
import math
import numpy as np

# refer : https://github.com/makalo/CornerNet
def gaussian_radius(det_size, min_overlap = 0.7):
    width, height = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2

    return min(r1, r2, r3)

def gaussian2D(shape, p, sigma):
    w, h = shape
    px, py = p[0], p[1]
    
    x = np.arange(w) + 0.5
    y = np.arange(h) + 0.5
    x, y = np.meshgrid(x, y)

    y = np.exp(-((x - px)**2 + (y - py)**2) / (2 * sigma * sigma))

    # exception
    # y[y < 0] = 0
    # y[y > 1] = 1

    return y

def generate_gaussian(points, radius):
    gaussian = gaussian2D((radius, radius), points, sigma = radius / 6.)
    return gaussian

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # case 1 - JSH
    fig = plt.figure()
    ax = fig.gca(projection = "3d")

    radius = gaussian_radius([200, 300])
    print(radius)
    input()

    gaussian = generate_gaussian([100, 150], radius)

    h, w = gaussian.shape

    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    
    ax.plot_surface(x, y, gaussian, cmap = cm.coolwarm, linewidth = 0, antialiased = False)

    plt.show()
