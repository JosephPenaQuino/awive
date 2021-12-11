'''Space Time Image Velocimetry'''

import json
import math
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
        self._ksize = self._config['ksize']
        # self._generate_st_images(config_path, video_identifier)
        self._stis.append(np.load('sti_00.npy'))

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

    def _process_sti(self, image: np.ndarray):
        '''process sti image'''
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self._ksize)
        sobelt = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self._ksize)

        Jxx = sum(sum(sobelx * sobelx))
        Jtt = sum(sum(sobelt * sobelt))
        Jxt = sum(sum(sobelx * sobelt))
        angle = 180 *  math.atan2(2*Jxt, Jtt - Jxx) / 2 / math.pi
        coherence = math.sqrt((Jtt-Jxx)**2 + 4*Jxt**2) / (Jxx + Jtt)
        return angle, coherence

    @staticmethod
    def _get_new_point(point, angle, length):
        '''
        point - Tuple (x, y)
        angle - Angle you want your end point at in degrees.
        length - Length of the line you want to plot.

        Will plot the line on a 10 x 10 plot.
        '''
        # unpack the first point
        x, y = point
        # find the end point
        rad_angle = math.radians(angle)
        endy = length * math.sin(rad_angle)
        endx = length * math.cos(rad_angle)
        return int(endx+x), int(-endy+y)

    def _get_image_with_line(self, image, angle):
        (width, height) = image.shape
        old_point = (int(width/2), int(height/2))
        new_point = self._get_new_point(old_point, angle, 30)
        cv2.line(image, new_point, old_point, 255, 1)
        return image

    def run(self):
        '''Execute'''
        window_width = self._config['window_shape'][0]
        window_height = self._config['window_shape'][1]
        for idx, sti in enumerate(self._stis):
            print(f'space time image {idx} shape: {sti.shape}')
            width = sti.shape[0]
            height = sti.shape[1]
            final_image = []
            wm = window_width//2
            hm = window_height//2
            for i in range(width//window_width):
                final_image.append([])
                for j in range(height//window_height):
                    s = [i*window_width,(i+1)*window_width]
                    e = [j*window_height,(j+1)*window_height]
                    image_window = sti[s[0]:s[1], e[0]:e[1]]
                    # cv2.imshow('image window', image_window)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    angle, coherence = self._process_sti(image_window)
                    print((f'- at ({i}, {j}): angle = {angle:0.2f}, '
                           f'coherence={coherence:0.2f}'))
                    new_image = self._get_image_with_line(
                            image_window,
                            angle
                            )
                    final_image[i].append(new_image)
                final_image[i] = np.hstack(final_image[i])
            final_image = np.vstack(final_image)

            # save and plot iamge
            np.save('stiv_final.npy', final_image)
            cv2.imshow('stiv final', final_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



def main(config_path: str, video_identifier: str):
    '''Basic example of STIV usage'''
    stiv = STIV(config_path, video_identifier)
    stiv.run()


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
