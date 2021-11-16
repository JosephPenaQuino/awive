'''Play  a video'''
import argparse

import cv2

from loader import get_loader, Loader
from correct_image import Formatter


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'
RESIZE_RATIO = 5


def play(loader: Loader, formatter: Formatter, undistort=True, roi=True):
    '''Plays a video'''

    while loader.has_images():
        image = loader.read()
        if undistort:
            image = formatter.apply_distortion_correction(image)
        if roi:
            image = formatter.apply_roi_extraction(image)
        lil_im = cv2.resize(image, (1000, 1000))
        cv2.imshow('Video', lil_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print ('Finished by key \'q\'')
            break
    cv2.destroyAllWindows()


def main(config_path: str, video_identifier: str, undistort=True, roi=True):
    '''Read configurations and play video'''
    loader = get_loader(config_path, video_identifier)
    formatter = Formatter(config_path, video_identifier)
    play(loader, formatter, undistort, roi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "statio_name",
        help="Name of the station to be analyzed")
    parser.add_argument(
        "video_identifier",
        help="Index of the video of the json config file")
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
         undistort=args.undistort,
         roi=args.roi
         )
