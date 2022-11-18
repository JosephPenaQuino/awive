"""Basic example using AWIVE."""

import cv2
import numpy as np
from numpy.typing import NDArray

from awive.loader import Loader, get_loader

CONFIG_PATH = "examples/basic/config.json"
VIDEO_ID = "basic"


def basic_plot_image(config_path: str, video_identifier: str) -> None:
    """Use basic loader functions to read an image."""
    loader: Loader = get_loader(config_path, video_identifier)
    print(f"{type(loader)=}")
    image: NDArray[np.uint8] = loader.read()
    print(f"{image.shape=}")
    print(f"{image.dtype=}")
    print(f"{image=}")
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    loader.end()


if __name__ == "__main__":
    basic_plot_image(CONFIG_PATH, VIDEO_ID)
