#!/home/joseph/anaconda3/envs/imageProcessing/bin/python3
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
        roi2  = config['roi2']
        self._wr0 = slice(roi2[0][0], roi2[1][0])
        self._wr1 = slice(roi2[0][1], roi2[1][1])
        self._r0 = slice(roi[0][0], roi[1][0])
        self._r1 = slice(roi[0][1], roi[1][1])
        self._roi_shape = (roi[1][0] - roi[0][0], roi[1][1] - roi[0][1])

        ksize = config['kernel_size']
        self._kernel = np.ones((ksize, ksize),np.uint8)

    def _get_difference_accumulation(self):
        cnt = 0
        buffer = []
        accumulated_image = np.zeros(self._roi_shape)

        image = self._loader.read()
        ref_image = image[self._wr0, self._wr1]
        np.save('im_ref.npy', ref_image)
        image = image[self._r0, self._r1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for i in range(self._buffer_length):
            if not self._loader.has_images():
                print('broke')
                return None
            new_image = self._loader.read()[self._r0, self._r1]
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

            accumulated_image += (new_image - image) ** 2
            image = new_image
            # buffer.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            # cnt +=1

        # for image in buffer[:-1]:
            # new_difference = abs(buffer[-1] - image)
            # np.save('i0.npy', buffer[-1])
            # np.save('i1.npy', image)
            # np.save('i2.npy', new_difference)
            # accumulated_image += new_difference
        # accumulated_image = np.interp(
            #     accumulated_image,
            #     (accumulated_image.min(), accumulated_image.max()),
            #     (0, 255)
            #     ).astype(np.uint8)
        np.save('acc.npy', accumulated_image)
        print('average:', round(accumulated_image.mean(), 2))

        return accumulated_image

    @staticmethod
    def _get_threshold(image):
        '''get threshold'''
        hist, _ = np.histogram(image.ravel(), density=True, bins=255)
        max_idx = 0
        max_slope = abs(hist[0] - hist[1])
        for i in range(len(hist)-1):
            new_slope = abs(hist[i] - hist[i-1])
            if new_slope > max_slope:
                max_slope = new_slope
                max_idx = i
        threshold = int(255 * (max_idx + 1) / len(hist))
        return threshold, hist

    def _compute_water_level(self, image, threshold):
        '''using given threshold compute the water level of the image'''
        print('threshold:', threshold)
        image = (image > threshold).astype(np.uint8)
        np.save('out1.npy', image)

        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self._kernel)
        total = image.shape[0] * image.shape[1]
        print('total', total)
        msum = image.sum()
        print('msum', msum)
        p = round(msum / total, 2)
        print('percentage:', p)
        height = image.shape[0] - int(p * image.shape[0])
        for i in range(image.shape[1]):
            image[height][i] = 3
        np.save('out2.npy', image)
        return height

    def get_water_level(self):
        '''calculate and return water level'''
        accumulated_image = self._get_difference_accumulation()
        np.save('out0.npy', accumulated_image)
        # threshold, _ = self._get_threshold(accumulated_image)
        # height = self._compute_water_level(accumulated_image, threshold)
        return 0

        # return height


def main(config_path: str, video_identifier: str, show_image=True):
    '''Execute basic example of water level detector'''
    water_level_detector = WaterlevelDetector(config_path, video_identifier)
    water_level = water_level_detector.get_water_level()
    # print('water level:', water_level)


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
