#!/home/joseph/anaconda3/envs/imageProcessing/bin/python3
'''Play  a video'''
import argparse
import numpy as np
import json

import cv2

from loader import get_loader, Loader
from correct_image import Formatter


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'
RESIZE_RATIO = 5


def play(loader: Loader, formatter: Formatter, undistort=True, roi=True, time_delay=1, resize=False, wlcrop=None):
    '''Plays a video'''
    i =0

    while loader.has_images():
        image = loader.read()
        if undistort:
            image = formatter.apply_distortion_correction(image)
        if roi:
            image = formatter.apply_roi_extraction(image)
        elif wlcrop is not None:
            image = image[wlcrop[0], wlcrop[1]]
        if resize:
            lil_im = cv2.resize(image, (1000, 1000))
        else:
            lil_im = image
        cv2.imshow('Video', lil_im)
        np.save(f'images/im_{i:04}.npy',lil_im)
        if cv2.waitKey(time_delay) & 0xFF == ord('q'):
            print ('Finished by key \'q\'')
            break
        i += 1
    cv2.destroyAllWindows()


def main(config_path: str, video_identifier: str, undistort=True, roi=True, time_delay=1, resize=True, wlcrop=True):
    '''Read configurations and play video'''
    loader = get_loader(config_path, video_identifier)
    formatter = Formatter(config_path, video_identifier)
    if wlcrop:
        with open(config_path) as json_file:
            config = json.load(json_file)[video_identifier]['water_level']
        roi2  = config['roi']
        wr0 = slice(roi2[0][0], roi2[1][0])
        wr1 = slice(roi2[0][1], roi2[1][1])
        crop = (wr0, wr1)
    else:
        crop = None
    play(loader, formatter, undistort, roi, time_delay, resize, crop)


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
    parser.add_argument(
        '-c',
        '--wlcrop',
        action='store_true',
        help='Water level crop')
    parser.add_argument(
        '-z',
        '--resize',
        action='store_true',
        help='Resizer image to 1000x1000')
    parser.add_argument(
        '-t',
        '--time',
        default=1,
        type=int,
        help='Time delay between each frame (ms)')
    args = parser.parse_args()
    CONFIG_PATH = f'{FOLDER_PATH}/{args.statio_name}.json'
    main(config_path=CONFIG_PATH,
         video_identifier=args.video_identifier,
         undistort=args.undistort,
         roi=args.roi,
         time_delay=args.time,
         resize=args.resize,
         wlcrop=args.wlcrop
         )
