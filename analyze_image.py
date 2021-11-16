'''Analyze image savig it as numpy file'''

import argparse
import numpy as np

from loader import get_loader
from correct_image import Formatter


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'


def main(config_path: str, video_identifier: str, entire_frame=False,
        undistort=True, roi=True):
    '''Save the first image as numpy file'''
    loader = get_loader(config_path, video_identifier)
    formatter = Formatter(config_path, video_identifier)
    image = loader.read()
    if entire_frame:
        formatter.show_entire_image()
    if undistort:
        image = formatter.apply_distortion_correction(image)
    if roi:
        image = formatter.apply_roi_extraction(image)
    np.save('tmp.npy', image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'statio_name',
        help='Name of the station to be analyzed')
    parser.add_argument(
        'video_identifier',
        help='Index of the video of the json config file')
    parser.add_argument(
        '-f',
        '--frame',
        action='store_true',
        help='Plot entire frame or not')
    parser.add_argument(
        '-u',
        '--undistort',
        action='store_true',
        help='Format image using distortion correction')
    parser.add_argument(
        '-r',
        '--roi',
        action='store_true',
        help='Format image using selecting only roi area')
    args = parser.parse_args()
    CONFIG_PATH = f'{FOLDER_PATH}/{args.statio_name}.json'
    main(config_path=CONFIG_PATH,
         video_identifier=args.video_identifier,
         entire_frame=args.frame,
         undistort=args.undistort,
         roi=args.roi)
