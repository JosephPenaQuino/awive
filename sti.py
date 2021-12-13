#!/home/joseph/anaconda3/envs/imageProcessing/bin/python3
'''Space Time Image Velocimetry'''

import json
import math
import argparse
import cv2
import numpy as np
from correct_image import Formatter
from loader import get_loader


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'


class STIV():
    '''Space Time Image Velocimetry'''
    def __init__(self, config_path: str, video_identifier: str, debug=False):
        with open(config_path) as json_file:
            root_config = json.load(json_file)[video_identifier]
            self._config = root_config['stiv']
        self._debug = debug
        # Shall be initialized later
        self._fps = None

        self._stis = []
        self._stis_qnt = len(self._config['lines'])
        self._ksize = self._config['ksize']
        self._generate_st_images(config_path, video_identifier)
        self._overlap = self._config['overlap']

        self._ppm = root_config['PPM']
        print('frames per second:', self._fps)
        print('pixels per minute', self._ppm)

    def _get_velocity(self, angle):
        '''
        Given STI pattern angle, calculate velocity
        - angle in radians
        '''
        velocity = math.tan(angle) * self._fps / self._ppm
        return velocity

    def _generate_st_images(self, config_path, video_identifier):
        # generate space time images
        loader = get_loader(config_path, video_identifier)
        formatter = Formatter(config_path, video_identifier)
        self._fps = loader.fps

        # initialize set of sti images
        for _ in range(self._stis_qnt):
            self._stis.append([])

        # generate all lines
        coordinates_list = self._config['lines']
        cnt=0
        while loader.has_images():
            image = loader.read()
            image = formatter.apply_distortion_correction(image)
            image = formatter.apply_roi_extraction(image)
            np.save(f'images/im_{cnt:04}.npy', image)
            cnt+=1

            for i, coordinates in enumerate(coordinates_list):
                start = coordinates['start']
                end = coordinates['end']
                row = image[start[0], start[1]:end[1]]
                self._stis[i].append(row)

        for i in range(self._stis_qnt):
            self._stis[i] = np.array(self._stis[i])
            np.save(f'sti_{i:04}.npy', self._stis[i])

    @property
    def stis(self):
        '''Return Space-Time image'''
        return self._stis

    def _process_sti(self, image: np.ndarray):
        '''process sti image'''
        image = cv2.medianBlur(image, 7)
        sobelx = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self._ksize)
        sobelt = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self._ksize)
        if sobelx.sum() == 0 and sobelt.sum() == 0:
            return 0, 0

        Jxx = (sobelx * sobelx).sum()
        Jtt = (sobelt * sobelt).sum()
        Jxt = (sobelx * sobelt).sum()
        angle =  math.atan2(2*Jxt, Jtt - Jxx) / 2
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
        endy = length * math.cos(angle)
        endx = length * math.sin(angle)
        return int(endx+x), int(-endy+y)

    def _draw_angle(self, image, angle, position):
        (width, height) = image.shape
        new_point = self._get_new_point(position, angle, 10)
        cv2.line(image, new_point, position, 255, 1)
        return image

    def run(self, show_image=False):
        '''Execute'''
        window_width = int(self._config['window_shape'][0]/2)
        window_height = int(self._config['window_shape'][1]/2)
        velocities = []
        for idx, sti in enumerate(self._stis):
            print(f'space time image {idx} shape: {sti.shape}')
            width = sti.shape[0]
            height = sti.shape[1]
            final_image = []
            angle_accumulated = 0
            c_total = 0

            # plot vectors
            mask = np.zeros(sti.shape)

            s = window_width
            i = 0
            while s + window_width < width:
                j = 0
                e = window_height
                while e + window_height < height:
                    ss = slice(s-window_width, s+window_width)
                    ee = slice(e-window_height, e+window_height)
                    image_window = sti[ss,ee]
                    angle, coherence = self._process_sti(image_window)
                    angle_accumulated += (angle * coherence)
                    c_total += coherence
                    if self._debug:
                        print((f'- at ({i}, {j}): angle = '
                               f'- in ({s}, {e}): angle = '
                               f'{math.degrees(angle):0.2f}, '
                               f'coherence={coherence:0.2f}, '
                               f'velocity={round(self._get_velocity(angle),2)}'))
                    mask = self._draw_angle(mask, angle, (e, s))
                    j+=1
                    e += int(self._overlap)
                i+=1
                s += int(self._overlap)

            mean_angle = angle_accumulated / c_total
            print("weighted mean angle:", round(math.degrees(mean_angle), 2))

            velocity = self._get_velocity(mean_angle)
            print("velocity", round(velocity, 2))
            velocities.append(velocity)

            # save and plot iamge
            final_image = self._stis[idx] + mask
            np.save(f'stiv_final_{idx:02}.npy', final_image)
            if show_image:
                cv2.imshow('stiv final', final_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        total = 0
        for vel in velocities:
            total += vel
        total /= len(velocities)
        print('Total mean velocity:', round(total, 2))



def main(config_path: str, video_identifier: str, show_image=True, debug=False):
    '''Basic example of STIV usage'''
    stiv = STIV(config_path, video_identifier, debug)
    stiv.run(show_image)


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
        '-d',
        '--debug',
        action='store_true',
        help='Activate debug mode')
    parser.add_argument(
        '-i',
        '--image',
        action='store_true',
        help='Show every space time image')
    args = parser.parse_args()
    CONFIG_PATH = f'{args.path}/{args.statio_name}.json'
    main(config_path=CONFIG_PATH,
         video_identifier=args.video_identifier,
         show_image=args.image,
         debug=args.debug
         )
