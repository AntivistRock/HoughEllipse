from PIL import Image, ImageFilter
import numpy as np
import cv2

from generalized_hough import hough_ellipse


def white_ellipse_on_black(h, w, x, y, angle, a, b):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.ellipse(img, (y, x), (a, b), angle, startAngle=0, endAngle=360,
                color=(255, 255, 255), thickness=-1)
    return Image.fromarray(img)


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


def test_basic():
    x, y, angle, a, b = 20, 10, 0, 6, 12
    img = white_ellipse_on_black(50, 50, x, y, angle, a, b)

    ellipses_params = hough_ellipse(img)
    assert contains_similar(ellipses_params, (x, y, angle, a - 5, b - 5))
    assert not contains_similar(ellipses_params,
                                (x + 5, y + 5, angle, a - 5, b - 5))


class TestDifferentImages:
    def test_special_sizes(self):
        for side in [32, 64, 31, 33, 16 * 3]:
            x, y, angle, a, b = side - 14, side - 9, 0, 6, 10
            img = white_ellipse_on_black(side, side, x, y, angle, a, b)

            params = hough_ellipse(img)
            assert contains_similar(params, (x, y, angle, a - 5, b - 5))
            assert not contains_similar(params,
                                        (x - 5, y - 5, angle, a - 5, b - 5))

    def test_small_image(self):
        x, y, angle, a, b = 5, 5, 0, 4, 4
        img = white_ellipse_on_black(10, 10, x, y, angle, a, b)
        ellipses_params = hough_ellipse(img)
        assert contains_similar(ellipses_params, (x, y, angle, a - 5, b - 5))

    def test_not_square_images(self):
        for (h, w) in [(20, 50), (10, 100)]:
            x, y, angle, a, b = 5, w - 16, 0, 12, 4
            img = white_ellipse_on_black(h, w, x, y, angle, a, b)

            params = hough_ellipse(img)
            assert contains_similar(params, (x, y, angle, a - 5, b - 5))
            assert not contains_similar(params,
                                        (x - 2, y - 5, angle, a - 5, b - 5))

            x, y, a, b = y, x, b, a
            img = white_ellipse_on_black(w, h, x, y, angle, a, b)

            params = hough_ellipse(img)
            assert contains_similar(params, (x, y, angle, a - 5, b - 5))
            assert not contains_similar(params,
                                        (x - 2, y - 5, angle, a - 5, b - 5))

    def test_one_channel(self):
        x, y, angle, a, b = 10, 20, 0, 12, 6

        img = np.zeros((50, 50), dtype=np.uint8)
        cv2.ellipse(img, (y, x), (a, b), angle, startAngle=0, endAngle=360,
                    color=255, thickness=-1)

        ellipses_params = hough_ellipse(Image.fromarray(img))
        assert contains_similar(ellipses_params, (x, y, angle, a - 5, b - 5))

    def test_coloured(self):
        x, y, angle, a, b = 10, 20, 0, 12, 6

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.ellipse(img, (y, x), (a, b), angle, startAngle=0, endAngle=360,
                    color=(0, 255, 0), thickness=-1)

        ellipses_params = hough_ellipse(Image.fromarray(img))
        assert contains_similar(ellipses_params, (x, y, angle, a - 5, b - 5))

    def test_four_channels(self):
        x, y, angle, a, b = 10, 20, 0, 12, 6

        img = np.zeros((50, 50, 4), dtype=np.uint8)
        cv2.ellipse(img, (y, x), (a, b), angle, startAngle=0, endAngle=360,
                    color=(100, 255, 100, 200), thickness=-1)

        ellipses_params = hough_ellipse(Image.fromarray(img))
        assert contains_similar(ellipses_params, (x, y, angle, a - 5, b - 5))


class TestDifferentEllipses:
    def test_no_ellipses(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        ellipses_params = hough_ellipse(Image.fromarray(img))
        assert len(ellipses_params) == 0

    def test_two_tangent(self):
        ellipses = [(20, 10, 0, 6, 12), (20, 24, 0, 6, 12)]
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        for x, y, angle, a, b in ellipses:
            cv2.ellipse(img, (y, x), (a, b), angle, startAngle=0, endAngle=360,
                        color=(255, 255, 255), thickness=-1)

        params = hough_ellipse(Image.fromarray(img))
        for x, y, angle, a, b in ellipses:
            assert contains_similar(params, (x, y, angle, a - 5, b - 5))
        assert not contains_similar(params, (20, 16, 0, 12, 10))


def test_blurry():
    x, y, angle, a, b = 10, 20, 0, 12, 6
    img = white_ellipse_on_black(50, 50, x, y, angle, a, b)
    blurred = img.filter(ImageFilter.GaussianBlur(2))

    ellipses_params = hough_ellipse(blurred)
    assert contains_similar(ellipses_params, (x, y, angle, a - 5, b - 5))
