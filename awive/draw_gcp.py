"""Draw a circle in the ground control point."""

import argparse
import json

import cv2
import numpy as np

from awive.correct_image import Formatter
from libs.npyplotter.npyplotter.plot_npy import picshow


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'
TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2


def draw_circle(img, center, number):
    """Draw a circle with a number."""
    text = str(number)
    cv2.circle(img, center, 10, 255, -1)

    text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    text_origin: tuple[int, int] = (
        int(center[0] - text_size[0] / 2),
        int(center[1] + text_size[1] / 2)
    )

    cv2.putText(
        img,
        text,
        text_origin,
        TEXT_FACE,
        TEXT_SCALE,
        0,
        TEXT_THICKNESS,
        cv2.LINE_AA
    )


def main(
    config_path: str,
    video_identifier: str,
    plot=False,
    undistort=False,
    roi=False
) -> None:
    """Read config and draw circles in the image."""
    with open(config_path) as json_file:
        config = json.load(json_file)[video_identifier]['gcp']
    formatter = Formatter(config_path, video_identifier)

    img = np.load('tmp.npy')

    pixels = config['pixels']

    le = len(pixels['x'])
    for i in range(le):
        draw_circle(img, (pixels['x'][i], pixels['y'][i]), i+1)

    if undistort:
        img = formatter.apply_distortion_correction(img)
    if roi:
        img = formatter.apply_roi_extraction(img)

    np.save('tmp2.npy', img)
    if plot:
        picshow([img])


if __name__ == '__main__':
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
        '-p',
        '--path',
        help='Path to the config folder',
        type=str,
        default=FOLDER_PATH
    )
    parser.add_argument(
        '-P',
        '--plot',
        action='store_true',
        help='Plot output image'
    )
    parser.add_argument(
        '-r',
        '--roi',
        action='store_true',
        help='Format image using selecting only roi area'
    )
    parser.add_argument(
        '-u',
        '--undistort',
        action='store_true',
        help='Format image using distortion correction'
    )
    args = parser.parse_args()
    main(
        config_path=f'{args.path}/{args.statio_name}.json',
        video_identifier=args.video_identifier,
        undistort=args.undistort,
        roi=args.roi,
        plot=args.plot
    )
