'''Loader of videos of frames'''

import os
import json
import abc
import numpy as np
import cv2


def get_loader(config_path, video_identifier):
    '''Return a ImageLoader or VideoLoader class
    Read config path and if the image dataset folder is full the function
    returns a ImageLoader, if not, then returns a VideoLoader
    '''
    # check if in image folder there are located the extracted images
    with open(config_path) as json_file:
        config = json.load(json_file)[video_identifier]

    image_folder_path = config['image_dataset']
    if any(os.scandir(image_folder_path)):
        return ImageLoader(config)
    return VideoLoader(config)


class Loader(metaclass=abc.ABCMeta):
    '''Abstract class of loader'''
    def __init__(self, offset):
        self._offset = offset

    @abc.abstractmethod
    def has_images(self):
        '''Check if the source contains one more frame'''

    @abc.abstractmethod
    def read(self):
        '''Read a new image from the source'''

    @abc.abstractmethod
    def end(self):
        '''Free all resources'''


class ImageLoader(Loader):
    '''Loader that loads images from a directory'''

    def __init__(self, config: dict):
        super().__init__(config['image_number_offset'])
        self._image_dataset = config['image_dataset']
        self._prefix = config['image_path_prefix']
        self._digits = config['image_path_digits']
        self._index = 0
        self._image_number = len(os.listdir(self._image_dataset))

    def has_images(self):
        return self._index < self._image_number

    def _path(self, i):
        i += self._offset
        if self._digits == 5:
            return f'{self._image_dataset}/{self._prefix}{i:05}.jpg'
        elif self._digits == 3:
            return f'{self._image_dataset}/{self._prefix}{i:03}.jpg'
        return f'{self._image_dataset}/{self._prefix}{i:04}.jpg'

    def set_index(self, index):
        '''Set index of the loader to read any image from the folder'''
        self._index = index

    def read(self):
        self._index += 1
        return cv2.imread(self._path(self._index))

    def end(self):
        pass


class VideoLoader(Loader):
    '''Loader that loads from a video'''

    def __init__(self, config: dict):
        super().__init__(config['image_number_offset'])
        self._video_path = config['video_path']
        self._cap = cv2.VideoCapture(self._video_path)
        self._image = None # Current image
        self._image_read = False # Check if the current images was read

        # Skip offset
        for _ in range(self._offset+1):
            if self.has_images():
                self.read()

    def has_images(self):
        if not self._cap.isOpened():
            return False
        ret, self._image = self._cap.read()
        self._image_read = False
        return ret

    def read(self):
        if self._image_read:
            _, self._image = self._cap.read()
        self._image_read = True
        return self._image

    def end(self):
        self._cap.release()
        pass


class Formatter:
    '''Format frames in order to be used by image processing methods'''

    def __init__(self, shape, grades, a, w1, w2, h1, h2, gray=True):
        self._width = shape[0]
        self._height = shape[1]
        self._grades = grades
        self._a = a
        self._M = cv2.getRotationMatrix2D((self._width//2, self._height//2),
                                         grades,
                                         a)
        self._w1 = w1
        self._w2 = w2
        self._h1 = h1
        self._h2 = h2
        self._gray = gray

    def apply(self, image: np.ndarray) -> np.ndarray:
        '''Apply format methods in current image'''
        # Rotate image
        if self._grades != 0:
            image = cv2.warpAffine(image, self._M, (self._width, self._height))
        # Crop image
        image = image[self._w1:self._w2, self._h1:self._h2]
        # To gray
        if self._gray:
            image = cv2.cv2tColor(image, cv2.COLOR_BGR2GRAY)
        return image
