from scipy.signal import convolve2d
import numpy as np
import math


def HoughEllipse(img):
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

    A = np.zeros((img_w_size, img_h_size))
    pairs = np.argwhere(magnitude > 100)  # edge pixels
    a = 12
    b = 5

    for x, y in pairs:
        dX = h_grad[x][y]
        dY = v_grad[x][y]

        angle = math.atan2(dY, dX) - math.pi / 2
        xi = math.tan(angle)
        # dx и dy поменяны местами, поскольку у нашего изображения поменяны оси
        dy = 0
        if xi != 0:
            dy = -np.sign(dX) * a / np.sqrt(1 + (b / (a * xi)) ** 2)
        dx = -np.sign(dY) * b / np.sqrt(1 + (a * xi / b) ** 2)

        x_0 = int(x + dx)
        y_0 = int(y + dy)
        if 0 < x_0 < img_w_size and 0 < y_0 < img_h_size:
            A[x_0, y_0] += 1

    ellipses_params = np.argwhere(A > 4)
    return ellipses_params
