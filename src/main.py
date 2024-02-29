import numpy as np
from PIL import Image

from generalized_hough import HoughEllipse
from utils.draw_ellipses import draw_ellips_bound

if __name__ == "__main__":
    img = Image.open("../image_examples/example_img.jpeg")
    ellipses_params = HoughEllipse(img)

    # Save params
    np.savetxt('../results/centers.txt', ellipses_params, fmt='%d')

    img = img.convert("RGB")
    for params in ellipses_params:
        x, y = params
        img = draw_ellips_bound(img, (y, x), 12, 5, 0)

    img.save('../results/detected_img.jpeg')