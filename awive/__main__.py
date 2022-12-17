"""Main application of Adaptive Water Image Velocimetry Estimator."""

import argparse

import cv2
import numpy as np
from numpy.typing import NDArray

from awive.loader import Loader, get_loader


DEFAULT_DATASET_PATH = "/home/joseph/Documents/Thesis/Dataset/config"


def main(config_path: str, video_identifier: str) -> None:
    """Use basic loader functions to read an image."""
    loader: Loader = get_loader(config_path, video_identifier)
    image: NDArray[np.uint8] = loader.read()
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    loader.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "statio_name",
        help="Name of the station to be analyzed"
    )
    parser.add_argument(
        "video_identifier",
        help="Index of the video of the json config file"
    )
    parser.add_argument(
        "--dataset_path",
        dest="dataset_path",
        default=DEFAULT_DATASET_PATH,
        help="Path to the dataset folder",
    )
    args = parser.parse_args()
    main(
        config_path=f"{args.dataset_path}/{args.statio_name}.json",
        video_identifier=args.video_identifier,
    )
