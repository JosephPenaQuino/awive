#!/home/joseph/anaconda3/envs/imageProcessing/bin/python3
'''Draw a circle in the ground control point'''

import os
import json
import argparse
import numpy as np
import cv2
from correct_image import Formatter

FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'
TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
TEXT_SCALE = 1.5
TEXT_THICKNESS = 2


def draw_circle(img, center, number):
    '''draw a circle with a number'''
    text = str(number)
    cv2.circle(img, center, 30, 255, -1)

    text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    text_origin = (int(center[0] - text_size[0] / 2), int(center[1] + text_size[1] / 2))

    cv2.putText(img, text, text_origin, TEXT_FACE, TEXT_SCALE, 0, TEXT_THICKNESS, cv2.LINE_AA)

def main(config_path: str, video_identifier: str, plot=False, undistort=False,
        roi=False):
    '''read config and draw circles in the image'''
    with open(config_path) as json_file:
        config = json.load(json_file)[video_identifier]['gcp']
    formatter = Formatter(config_path, video_identifier)

    img = np.load('tmp.npy')

    pixels = config['pixels']

    for i in range(4):
        draw_circle(img, (pixels['x'][i], pixels['y'][i]), i+1)

    if undistort:
        img = formatter.apply_distortion_correction(img)
    if roi:
        img = formatter.apply_roi_extraction(img)

    np.save('tmp2.npy', img)
    if plot:
        os.system('plotNpy tmp2.npy')

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "statio_name",
        help="Name of the station to be analyzed")
    parser.add_argument(
        "video_identifier",
        help="Index of the video of the json config file")
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the config folder',
        type=str,
        default=FOLDER_PATH)
    parser.add_argument(
        '-P',
        '--plot',
        action='store_true',
        help='Plot output image')
    parser.add_argument(
        '-r',
        '--roi',
        action='store_true',
        help='Format image using selecting only roi area')
    parser.add_argument(
        '-u',
        '--undistort',
        action='store_true',
        help='Format image using distortion correction')
    args = parser.parse_args()
    CONFIG_PATH = f'{args.path}/{args.statio_name}.json'
    main(config_path=CONFIG_PATH,
         video_identifier=args.video_identifier,
         undistort=args.undistort,
         roi=args.roi,
         plot=args.plot
         )
