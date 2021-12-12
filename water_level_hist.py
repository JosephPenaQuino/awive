'''
Water Level Detector using method described in:
Embedded implementation of image-based water-level measurement system
by:
- Kim, J.
- Han, Y.
- Hahn, H.
'''

import json
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
from loader import get_loader


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'


class WaterlevelDetector:
    '''Water Level Detector'''
    def __init__(self, config_path: str, video_identifier: str):
        with open(config_path) as json_file:
            config = json.load(json_file)[video_identifier]['water_level']
        self._loader = get_loader(config_path, video_identifier)
        self._buffer_length = config['buffer_length']
        roi  = config['roi']
        self._r0 = slice(roi[0][0], roi[1][0])
        self._r1 = slice(roi[0][1], roi[1][1])
        self._roi_shape = (roi[1][0] - roi[0][0], roi[1][1] - roi[0][1])

    def _get_difference_accumulation(self, show_image= False):
        cnt = 0
        buffer = []

        for _ in range(self._buffer_length):
            if not self._loader.has_images():
                print('broke')
                return None
            image = self._loader.read()[self._r0, self._r1]
            buffer.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            cnt +=1
        accumulated_image = np.zeros(self._roi_shape)

        for image in buffer[:-1]:
            new_difference = abs(buffer[-1] - image)
            np.save('i0.npy', buffer[-1])
            np.save('i1.npy', image)
            np.save('i2.npy', new_difference)
            accumulated_image += new_difference


        if show_image:
            np.save('roi_image.npy', accumulated_image)
            plt.hist(accumulated_image.ravel(), density=True,  bins=10)
            plt.ylabel('Probability')
            plt.xlabel('Data')
            plt.show()
            cv2.imshow('roi', accumulated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return accumulated_image

    def get_water_level(self, show_image=False):
        '''calculate and return water level'''
        image = self._get_difference_accumulation(show_image)
        return 0


def main(config_path: str, video_identifier: str, show_image=True):
    '''Execute basic example of water level detector'''
    water_level_detector = WaterlevelDetector(config_path, video_identifier)
    water_level = water_level_detector.get_water_level(show_image)
    print('water level:', water_level)


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
    parser.add_argument(
        '-i',
        '--image',
        action='store_true',
        help='Show every space time image')
    args = parser.parse_args()
    CONFIG_PATH = f'{args.path}/{args.statio_name}.json'
    main(config_path=CONFIG_PATH,
         video_identifier=args.video_identifier,
         show_image=args.image
         )
