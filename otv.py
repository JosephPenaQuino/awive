'''Optical Tracking Image Velocimetry'''

import argparse
import json
import math
import numpy as np
import cv2
from correct_image import Formatter
from loader import (get_loader, Loader)
# from sti import applyMean


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'

def get_magnitude(kp1, kp2):
    '''get the distance between two keypoints'''
    return math.sqrt((kp2.pt[1] - kp1.pt[1])**2 + (kp2.pt[0] - kp1.pt[0]) ** 2 )

def get_angle(kp1, kp2):
    '''get angle between two key points'''
    return math.atan2(kp2.pt[1] - kp1.pt[1] , kp2.pt[0] - kp1.pt[0]) *  180 / math.pi

def _get_velocity(kp1, kp2, real_distance_pixel, time, fps):
    return get_magnitude(kp1, kp2) * real_distance_pixel * fps / time





class OTV():
    '''Optical Tracking Image Velocimetry'''
    def __init__(self, config_path: str, video_identifier: str,
            prev_gray: np.ndarray):
        with open(config_path) as json_file:
            config = json.load(json_file)[video_identifier]['otv']
        self.feature_params = {
            'maxCorners': config['features']['maxcorner'],
            'qualityLevel': config['features']['qualitylevel'],
            'minDistance': config['features']['mindistance'],
            'blockSize': config['features']['blocksize']
            }
        self._partial_max_angle = config['partial_max_angle']
        self._partial_min_angle = config['partial_min_angle']
        self._final_max_angle = config['final_max_angle']
        self._final_min_angle = config['final_min_angle']
        self._final_min_distance = config['final_min_distance']
        self._max_features = config['max_features']
        self._radius = config['lk']['radius']
        self._max_level = config['lk']['max_level']
        self._resolution = config['resolution']

        winsize = config['lk']['winsize']

        self.lk_params = {
            'winSize': (winsize, winsize),
            'maxLevel': self._max_level,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         10,
                         0.03)
            }
        self.prev_gray = prev_gray
        self.prev = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=None,
            **self.feature_params)

    def _partial_filtering(self, kp1, kp2, max_distance):
        magnitude = get_magnitude(kp1, kp2)  #only to limit the research window
        if magnitude > max_distance:
            return False
        angle = get_angle(kp1, kp2)
        if self._partial_min_angle < angle < self._partial_max_angle:
            return True
        return False

    def _final_filtering(self, kp1, kp2):
        '''Final filter of keypoints'''
        magnitude = get_magnitude(kp1, kp2)
        if magnitude < self._final_min_distance:
            return False
        angle = get_angle(kp1, kp2)
        if self._final_min_angle < angle < self._final_max_angle:
            return True
        return False

    def calc(self, gray):
        '''Calculate velocity vectors of OTV'''
        self.prev = cv2.goodFeaturesToTrack(
            self.prev_gray,
            mask=None,
            **self.feature_params)
        next_, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                    gray,
                                                    self.prev,
                                                    None,
                                                    **self.lk_params)
        good_old = self.prev[status == 1].astype(int)
        good_new = next_[status == 1].astype(int)
        self.prev_gray = gray.copy()
        self.prev = good_new.reshape(-1, 1, 2)
        return good_old, good_new

    def run(self, loader: Loader, formatter: Formatter):
        '''Execute OTV and get velocimetry'''
        detector = cv2.FastFeatureDetector_create()
        previous_frame = None
        keypoints_current = []
        keypoints_start = []
        time = []
        valid = []
        velocity_mem = []
        path = []
        velocity = []
        angle = []
        distance = []
        keypoints_predicted = []
        masks = []

        # Initialization
        print('loader total frames:', loader.total_frames)
        for i in range(loader.total_frames):
            valid.append([])
            velocity_mem.append([])
            velocity.append([])
            angle.append([])
            distance.append([])
            path.append([])

        while loader.has_images():
            current_frame = loader.read()
            current_frame = formatter.apply_distortion_correction(current_frame)
            current_frame = formatter.apply_roi_extraction(current_frame)
            # get features
            keypoints = detector.detect(current_frame, None)

            # update lot of lists
            if previous_frame is None:
                for i, keypoint in enumerate(keypoints):
                    if len(keypoints_current) < self._max_features:
                        keypoints_current.append(keypoint)
                        keypoints_start.append(keypoint)
                        time.append(loader.index)
                        valid[loader.index].append(False)
                        velocity_mem[loader.index].append(0)
                        path[loader.index].append(i)
            else:
                for i, keypoint in enumerate(reversed(keypoints)):
                    if len(keypoints_current) < self._max_features:
                        keypoints_current.append(keypoint)
                        keypoints_start.append(keypoint)
                        time.append(loader.index)
                        valid[loader.index].append(False)
                        velocity_mem[loader.index].append(0)

            if previous_frame is not None:
                status = []
                errors = []
                pts1 = cv2.KeyPoint_convert(keypoints_current)
                pts2, st, err = cv2.calcOpticalFlowPyrLK(
                    previous_frame,
                    current_frame,
                    pts1,
                    None,
                    **self.lk_params
                    )
                keypoints_predicted.clear()
                for pt2 in pts2:
                    keypoints_predicted.append(cv2.KeyPoint(pt2[0], pt2[1], 1.0))
                pts2 = pts2.astype(int)
                pts1 = pts1.astype(int)

                if pts2 is not None:
                    good_new = pts2[(st==1).T[0]]
                    good_old = pts1[(st==1).T[0]]

                max_distance = self._max_level * (2 * self._radius + 1)
                max_distance /= self._resolution

                k = 0

                for i, keypoint in enumerate(keypoints_current):
                    partial_filter = self._partial_filtering(
                        keypoint,
                        keypoints_predicted[i],
                        max_distance
                        )
                    if not st[i] or not partial_filter:
                        # print('passed partial filter')
                        final_filter = self._final_filtering(
                            keypoints_start[i],
                            keypoints_current[i]
                            )
                        if final_filter:
                            print('passed final filter')
                            velocity_i = _get_velocity(
                                keypoints_start[i],
                                keypoints_current[i],
                                self._pixel_to_real / self._resolution,
                                loader.index - time[i],
                                self._fps
                                )
                            angle_i = _get_angle(
                                keypoints_start[i],
                                keypoints_current[i]
                                )
                            module_start = keypoints_start[i].pt[0] / self._step
                            module_current = keypoints_start[i].pt[0] / self._step
                            if module_start == module_current:
                                subregion_velocity[module_start].append(velocity_i)
                                subregion_trajectories[module_start] += 1

                            # update storage
                            pos = loader.index.copy()

                    keypoints_current[k] = keypoints_current[i]
                    keypoints_start[k] = keypoints_start[i]
                    keypoints_predicted[k] = keypoints_predicted[i]
                    # print('loader index:', loader.index)
                    # print('path length:', len(path))
                    path[loader.index].append(i)
                    velocity_mem[loader.index].append(0)
                    valid[loader.index].append(False)
                    time[k] = time[i]
                    k += 1

                print(loader.index)


            if previous_frame is not None:
                output = draw_vectors(current_frame, good_new, good_old, masks)
                cv2.imshow("sparse optical flow", output)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            previous_frame = current_frame.copy()
        loader.end()
        cv2.destroyAllWindows()


def draw_vectors(image, new_list, old_list, masks):
    '''Draw vectors of velocity and return the output and update mask'''
    if len(image.shape) == 3:
        color = (0, 255, 0)
        thick = 2
    else:
        color = 255
        thick = 1

    # create new mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    for new, old in zip(new_list, old_list):
        mask = cv2.line(mask, new.ravel(), old.ravel(), color, thick)

    # update masks list
    masks.append(mask)
    if len(masks) < 3:
        return np.zeros(image.shape)
    if len(masks) > 3:
        masks.pop(0)


    # generate image with mask
    total_mask = np.zeros(mask.shape, dtype=np.uint8)
    for mask_ in masks:
        total_mask = cv2.add(total_mask, mask_)
    output = cv2.add(image, total_mask)
    return output



def main(config_path: str, video_identifier: str):
    '''Basic example of OTV'''
    loader = get_loader(config_path, video_identifier)
    formatter = Formatter(config_path, video_identifier)
    loader.has_images()
    image = loader.read()
    prev_gray = formatter.apply_roi_extraction(image)
    otv = OTV(config_path, video_identifier, prev_gray)
    otv.run(loader, formatter)

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
    main(config_path=CONFIG_PATH,
         video_identifier=args.video_identifier,
         )
