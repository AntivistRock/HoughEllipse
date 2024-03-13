from PIL import Image
import numpy as np
import cv2

from generalized_hough import hough_ellipse


CV_WHITE = (255, 255, 255)


def black_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


def are_similar(params1, params2, eps=1):
    for param1, param2 in zip(params1, params2):
        if abs(param2 - param1) > eps:
            return False
    return True


def contains_similar(params_estimated, true_params):
    for params in params_estimated:
        if are_similar(true_params, params):
            return True
    return False


def test_easy():
    x, y, angle, a, b = 20, 10, 0, 12, 6

    img = black_image(50, 50)
    cv2.ellipse(img, (y, x), (a, b), angle, startAngle=0, endAngle=360,
                color=CV_WHITE, thickness=-1)

    ellipses_params = hough_ellipse(Image.fromarray(img))
    assert contains_similar(ellipses_params, (x, y, angle, a - 5, b - 5))
    assert not contains_similar(ellipses_params,
                                (x + 5, y + 5, angle, a - 5, b - 5))
