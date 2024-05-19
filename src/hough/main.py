import click
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .generalized_hough import hough_ellipse
from .utils.draw_ellipses import draw_ellipse_boundary


@click.command()
@click.option(
    "--img_path", default="image_examples/example_img.jpeg", help="Path to image."
)
@click.option("--res_folder", default="results", help="Path to folder to save results.")
def main(img_path: str, res_folder: str) -> None:
    img = Image.open(img_path)
    ellipses_params = hough_ellipse(img)

    # Save params
    np.savetxt(res_folder + "/centres.txt", ellipses_params, fmt="%d")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title("Original image")
    ax[0].axis("off")
    ax[0].imshow(img)

    img = img.convert("RGB")
    for params in ellipses_params:
        x, y, theta, a, b = params
        img = draw_ellipse_boundary(img, (y, x), a + 5, b + 5, theta)

    ax[1].set_title("Result image")
    ax[1].axis("off")
    ax[1].imshow(img)
    fig.suptitle("Ellipses detection using general Hough transform")
    fig.savefig(res_folder + "/report.png")
    img.save(res_folder + "detected_img.jpeg")
