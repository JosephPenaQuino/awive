"""Analyze image savig it as numpy file."""

import argparse

import cv2
import matplotlib.pyplot as plt
from npyplotter.plot_npy import picshow
import numpy as np

from awive.correct_image import Formatter
from awive.loader import Loader, get_loader

FOLDER_PATH = "examples/datasets"


def main(
    config_path: str,
    video_identifier: str,
    entire_frame=False,
    undistort=True,
    roi=True,
    get_frame=True,
    plot=False
) -> None:
    """Save the first image as numpy file."""
    loader: Loader = get_loader(config_path, video_identifier)
    formatter = Formatter(config_path, video_identifier)
    image: np.ndarray = loader.read()
    if get_frame:
        cv2.imwrite("image.png", image)
    if entire_frame:
        formatter.show_entire_image()
    if undistort:
        image = formatter.apply_distortion_correction(image)
    if roi:
        image = formatter.apply_roi_extraction(image)
    if get_frame:
        cv2.imwrite("image.png", image)
    np.save("tmp.npy", image)
    if plot:
        print("Plotting image")
        picshow([image])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="python -m awive.analyze_image river-brenta d0000 -P",
        description=(
            "Analyze image savig it as numpy file.\n"
            "Order of processing:\n"
            " 1. Crop image using gcp.pixels parameter\n"
            " 2. If enabled, lens correction using preprocessing.image_correction\n"
            " 3. Orthorectification using relation gcp.pixels and gcp.real\n"
            " 4. Pre crop\n"
            " 5. Rotation\n"
            " 6. Crop\n"
            " 7. Convert to gray scale\n"
        ),
    )
    parser.add_argument(
        "station_name",
        help="Name of the station to be analyzed",
    )
    parser.add_argument(
        "video_identifier",
        help="Index of the video of the json config file",
    )
    parser.add_argument(
        "-f",
        "--frame",
        action="store_true",
        help="Plot entire frame or not")
    parser.add_argument(
        "-u",
        "--undistort",
        action="store_true",
        help="Format image using distortion correction")
    parser.add_argument(
        "-g",
        "--getframe",
        action="store_true",
        help="Get first frame")
    parser.add_argument(
        "-r",
        "--roi",
        action="store_true",
        help="Format image using selecting only roi area")
    parser.add_argument(
        "-P",
        "--plot",
        action="store_true",
        help="Plot output image")
    args = parser.parse_args()
    CONFIG_PATH = f"{FOLDER_PATH}/{args.station_name}/config.json"
    main(
        config_path=CONFIG_PATH,
        video_identifier=args.video_identifier,
        entire_frame=args.frame,
        undistort=args.undistort,
        roi=args.roi,
        get_frame=args.getframe,
        plot=args.plot
    )
