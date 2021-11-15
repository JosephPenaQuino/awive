'''Correct distortion of videos

This module contains classes and functions needed to correct distortion of
videos, either intrinsic or extring to the camera. Also it saves the corrected
frames in defined directory path.

'''

import json
import argparse
import numpy as np
import cv2
from loader import get_loader


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'


class Formatter:
    '''Format frames in order to be used by image processing methods'''

    def __init__(self, config_path: str, video_identifier: str):
        # read configuration file
        with open(config_path) as json_file:
            self._config = json.load(json_file)[video_identifier]

        sample_image = self._get_sample_image(config_path, video_identifier)
        self._shape = (sample_image.shape[0], sample_image.shape[1])

        self._grades = self._config['rotate_image']
        self._rotation_matrix = self._get_rotation_matrix()

        w_slice = slice(self._config['roi']['w1'],
                        self._config['roi']['w2'])
        h_slice = slice(self._config['roi']['h1'],
                        self._config['roi']['h2'])
        self._slice = (w_slice, h_slice)

    def _get_rotation_matrix(self):
        a = 1.0   # TODO: idk why is 1.0
        return cv2.getRotationMatrix2D(
            (self._shape[0]//2, self._shape[1]//2),
            self._grades,
            a)

    @staticmethod
    def _get_sample_image(config_path: str, vid_identifier: str) -> np.ndarray:
        loader = get_loader(config_path, vid_identifier)
        image = loader.read()
        loader.end()
        return image

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def show_entire_image(self):
        '''Set slice to cover the entire image'''
        w_slice = slice(0, 6000)
        h_slice = slice(0, 6000)
        self._slice = (w_slice, h_slice)

    def _rotate(self, image: np.ndarray) -> np.ndarray:
        if self._grades != 0:
            return cv2.warpAffine(image,
                                  self._rotation_matrix,
                                  self._shape)
        return image

    def _crop(self, image: np.ndarray) -> np.ndarray:
        return image[self._slice[0], self._slice[1]]

    def apply_roi_extraction(self, image: np.ndarray, gray=True) -> np.ndarray:
        '''Apply image rotation, cropping and rgb2gray'''
        image = self._rotate(image)
        image = self._crop(image)
        if gray:
            image = self._gray(image)
        return image

    def apply_distortion_correction(self, image: np.ndarray) ->np.ndarray:
        '''Given GCP, undistort image'''
        return image


def main(config_path: str, video_identifier: str):
    '''Demonstrate basic example of video correction'''
    loader = get_loader(config_path, video_identifier)
    formatter = Formatter(config_path, video_identifier)
    image = loader.read()
    image = formatter.apply_roi_extraction(image)
    image = formatter.apply_distortion_correction(image)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    loader.end()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "statio_name",
        help="Name of the station to be analyzed")
    parser.add_argument(
        "video_identifier",
        help="Index of the video of the json config file")
    args = parser.parse_args()
    CONFIG_PATH = f'{FOLDER_PATH}/{args.statio_name}.json'
    main(config_path=CONFIG_PATH, video_identifier=args.video_identifier)
