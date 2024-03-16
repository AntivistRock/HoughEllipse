from itertools import product
from scipy.signal import convolve2d
import numpy as np
import math


def hough_ellipse(img):
    """
    Функция позволяет находить на изображении эллипсы, параллельные
    горизонтальной оси с параметрами a=12, b=5.
    :return: центры найденных эллипсов
    """

    gimg = img.convert(mode='L')

    G_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    G_y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

    h_grad = convolve2d(gimg, G_x)
    v_grad = convolve2d(gimg, G_y)

    magnitude = np.sqrt(np.power(h_grad, 2) + np.power(v_grad, 2))

    img_w_size = magnitude.shape[0]
    img_h_size = magnitude.shape[1]
    theta_range = range(0, 11)
    a_range = range(4, 16)
    b_range = range(4, 16)

    theta_size = len(theta_range)
    a_size = len(a_range)
    b_size = len(b_range)

    A = np.zeros((img_w_size, img_h_size, theta_size, a_size, b_size))
    pairs = np.argwhere(magnitude > 200)  # edge pixels

    for (x, y), theta, a, b in product(pairs, theta_range, a_range, b_range):
        dX = h_grad[x][y]
        dY = v_grad[x][y]

        theta_rad = math.radians(theta)

        angle = math.atan2(dY, dX) - theta_rad - math.pi / 2
        xi = math.tan(angle)
        # dx and dy swapped because image axes of an image-array are different.
        dy = 0
        if xi != 0:
            dy = -np.sign(dX) * a / np.sqrt(1 + (b / (a * xi)) ** 2)
        dx = -np.sign(dY) * b / np.sqrt(1 + (a * xi / b) ** 2)

        RotationMatrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                   [np.sin(theta_rad), np.cos(theta_rad)]])
        # Rotate by angle theta
        dx, dy = RotationMatrix @ np.array([dx, dy])

        x_0 = int(x + dx)
        y_0 = int(y + dy)

        if 0 < x_0 < img_w_size and 0 < y_0 < img_h_size:
            A[x_0, y_0, theta, a-5, b-5] += 1

    ellipses_params = np.argwhere(A > 10)
    return ellipses_params
