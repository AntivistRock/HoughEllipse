import numpy as np

from PIL import Image
import cv2


def draw_ellips_bound(img, center, a, b, angle):
    img = np.array(img)
    cv2.ellipse(img, center, (a, b), angle, 0, 360, (255, 0, 0), 1)
    return Image.fromarray(img)
