from PIL import Image

from hough.generalized_hough import hough_ellipse
from hough.utils.draw_ellipses import draw_ellipse_boundary


def test_basic(tmp_path):
    img = Image.open("image_examples/example_img.jpeg")
    ellipses_params = hough_ellipse(img)

    assert len(ellipses_params) > 0

    img = img.convert("RGB")
    for params in ellipses_params:
        x, y, theta, a, b = params
        img = draw_ellipse_boundary(img, (y, x), a + 5, b + 5, theta)

    img.save(tmp_path / "detected_img.jpeg")
    assert len(list(tmp_path.iterdir())) == 1
    assert Image.open(tmp_path / "detected_img.jpeg").size == img.size
