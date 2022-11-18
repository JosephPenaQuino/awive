"""Basic usage of Loader: Load an image and plot."""

from pathlib import Path

import cv2
import numpy as np
import requests as req
from numpy.typing import NDArray

from awive.loader import Loader, get_loader

CONFIG_PATH = "examples/basic/config.json"
VIDEO_PATH = "examples/basic/AlpineStabilised.avi"
VIDEO_ID = "basic"
FILE_ID = "1JreGYQEYUB4DkIk-MkE4n_-2RzSb0W27"


def download_basic_video(file_id: str, video_path: str) -> None:
    """Download basic video."""
    url: str = f"https://docs.google.com/uc?id={file_id}"
    print(f"{url=}")
    data = req.get(url)
    with open(video_path, 'wb')as file:
        file.write(data.content)


def basic_plot_image(config_path: str, video_identifier: str) -> None:
    """Use basic loader functions to read an image."""
    loader: Loader = get_loader(config_path, video_identifier)
    if not loader.has_images():
        raise ValueError("The video does not have images")
    image: NDArray[np.uint8] = loader.read()
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    loader.end()


if __name__ == "__main__":
    if not Path(VIDEO_PATH).exists():
        download_basic_video(FILE_ID, VIDEO_PATH)
    basic_plot_image(CONFIG_PATH, VIDEO_ID)
