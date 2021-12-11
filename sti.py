'''Space Time Image Velocimetry'''

import json
import argparse
import cv2
import numpy as np
from correct_image import Formatter
from loader import (get_loader, Loader)


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'


class STIV():
    '''Space Time Image Velocimetry'''
    def __init__(self, config_path: str, video_identifier: str):
        with open(config_path) as json_file:
            self._config = json.load(json_file)[video_identifier]['stiv']
        self._stis = []
        self._stis_qnt = len(self._config['lines'])
        self._generate_st_images(config_path, video_identifier)

    def _generate_st_images(self, config_path, video_identifier):
        # initialize set of sti images
        for _ in range(self._stis_qnt):
            self._stis.append([])

        # generate space time images
        loader = get_loader(config_path, video_identifier)
        formatter = Formatter(config_path, video_identifier)

        # generate all lines
        print('Generating stis images...')
        while loader.has_images():
            image = loader.read()
            image = formatter.apply_distortion_correction(image)
            image = formatter.apply_roi_extraction(image)

            coordinates_list = self._config['lines']
            for i, coordinates in enumerate(coordinates_list):
                start = coordinates['start']
                end = coordinates['end']
                row = image[start[0]:end[0], start[1]:end[1]]
                self._stis[i].append(row.ravel())

        for i in range(self._stis_qnt):
            self._stis[i] = np.array(self._stis[i])

    @property
    def stis(self):
        '''Return Space-Time image'''
        return self._stis

    def run(self, loader: Loader, formatter: Formatter):
        '''Execute'''


def main(config_path: str, video_identifier: str):
    '''Basic example of STIV usage'''
    stiv = STIV(config_path, video_identifier)
    stiv.run(loader, formatter)


if __name__ == '__main__':
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
    args = parser.parse_args()
    CONFIG_PATH = f'{args.path}/{args.statio_name}.json'
    main(config_path=CONFIG_PATH,
         video_identifier=args.video_identifier,
         )
