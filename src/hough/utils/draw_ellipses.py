import numpy as np

from PIL import Image
import cv2


def draw_ellipse_boundary(img: Image.Image, center: tuple[int, int], a: int, b: int, angle: float) -> Image.Image:
    """Draws ellipse on a given image.

    Args:
        img: Image to write ellipse on.
        center: (x, y) - position of the center of the circle on the image
        a, b: ellipse params
        angle: angle in radians to rotate ellips

    Returns:
        An image with circle.
    """
    img = np.array(img)
    cv2.ellipse(img, center, (a, b), angle, 0, 360, (255, 0, 0), 1)
    return Image.fromarray(img)
