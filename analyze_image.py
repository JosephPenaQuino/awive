'''Analyze image savig it as numpy file'''

import argparse
import json
import numpy as np

from loader import get_loader
from loader import Formatter


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'


def main(config_path: str, video_identifier: str, entire_frame=False):
    '''Save the first image as numpy file'''
    loader = get_loader(config_path, video_identifier)
    with open(config_path) as json_file:
        conf = json.load(json_file)[video_identifier]

    # Set formatter
    image = loader.read()
    if entire_frame:
        formatter = Formatter(image.shape,
                              conf['rotate_image'],
                              1.0,
                              0,
                              6000,
                              0,
                              6000,
                              gray=False)
    else:
        formatter = Formatter(image.shape,
                              conf['rotate_image'],
                              1.0,
                              conf['roi']['w1'],
                              conf['roi']['w2'],
                              conf['roi']['h1'],
                              conf['roi']['h2'],
                              gray=False)
    image = formatter.apply(image)
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
    args = parser.parse_args()
    CONFIG_PATH = f'{FOLDER_PATH}/{args.statio_name}.json'
    main(config_path=CONFIG_PATH,
         video_identifier=args.video_identifier,
         entire_frame=args.frame)
