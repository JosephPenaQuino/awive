#!/home/joseph/anaconda3/envs/imageProcessing/bin/python3
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

def compute_stats(velocity):
    avg = 0
    max_ = 0
    min_ = 100000000
    std_dev=0
    count = 0
    for i in range(len(velocity)):
        for j in range(len(velocity[i])):
            avg += velocity[i][j]
            max_ = velocity[i][j] if velocity[i][j] > max_ else max_
            min_ = velocity[i][j] if velocity[i][j] < min_ else min_
            count += 1

    if count  > 0:
        avg /= count

    for i in range(len(velocity)):
        for j in range(len(velocity[i])):
            std_dev += (velocity[i][j] - avg) ** 2
    if count > 0:
        std_dev = math.sqrt(std_dev/count)

    return avg, max_, min_, std_dev, count




class OTV():
    '''Optical Tracking Image Velocimetry'''
    def __init__(self, config_path: str, video_identifier: str,
            prev_gray: np.ndarray):
        with open(config_path) as json_file:
            root_config = json.load(json_file)[video_identifier]
            config = root_config['otv']
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
        self._step = config['region_step']
        self._resolution = config['resolution']
        self._pixel_to_real = config['pixel_to_real']

        width = root_config['roi']['w2'] - root_config['roi']['w1']
        height = root_config['roi']['h2'] - root_config['roi']['h1']
        self._mask = cv2.imread(config['mask_path'], 0) > 1
        self._mask = cv2.resize(
            self._mask.astype(np.uint8),
            (height, width),
            cv2.INTER_NEAREST
            )

        winsize = config['lk']['winsize']

        self.lk_params = {
            'winSize': (winsize, winsize),
            'maxLevel': self._max_level,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         config['lk']['max_count'],
                         config['lk']['epsilon'])
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

    def _apply_mask(self, image):
        if self._mask is not None:
            image = image * self._mask
        return image

    def _init_subregion_list(self, dimension, width):
        ret = []
        n_regions = math.ceil(width*self._resolution / self._step)
        for _ in range(n_regions):
            # TODO: This is so inneficient
            if dimension == 1:
                ret.append(0)
            elif dimension == 2:
                ret.append([])
        return ret


    def run(self, loader: Loader, formatter: Formatter, show_video=False):
        '''Execute OTV and get velocimetry'''
        # initialze parametrers
        detector = cv2.FastFeatureDetector_create()
        previous_frame = None
        keypoints_current = []
        keypoints_start = []
        time = []
        keypoints_predicted = []
        masks = []

        valid = []
        velocity_mem = []
        keypoints_mem_current = []
        keypoints_mem_predicted = []
        velocity = []
        angle = []
        distance = []
        path = []

        subregion_velocity = self._init_subregion_list(2, loader.width)
        subregion_trajectories = self._init_subregion_list(1, loader.width)

        # Initialization
        for i in range(loader.total_frames):
            valid.append([])
            velocity_mem.append([])
            velocity.append([])
            angle.append([])
            distance.append([])
            path.append([])

        while loader.has_images():
            # get current frame
            current_frame = loader.read()
            current_frame = formatter.apply_distortion_correction(current_frame)
            current_frame = formatter.apply_roi_extraction(current_frame)
            current_frame = self._apply_mask(current_frame)

            # get features as a list of KeyPoints
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

            print('Analyzing frame:', loader.index)
            if previous_frame is not None:
                pts1 = cv2.KeyPoint_convert(keypoints_current)
                pts2, st, _ = cv2.calcOpticalFlowPyrLK(
                    previous_frame,
                    current_frame,
                    pts1,
                    None,
                    **self.lk_params
                    )

                # add predicted by Lucas-Kanade new keypoints
                keypoints_predicted.clear()
                for pt2 in pts2:
                    keypoints_predicted.append(cv2.KeyPoint(
                        pt2[0],
                        pt2[1],
                        1.0
                        ))

                max_distance = self._max_level * (2 * self._radius + 1)
                max_distance /= self._resolution

                k = 0

                for i, keypoint in enumerate(keypoints_current):
                    partial_filter = self._partial_filtering(
                        keypoint,
                        keypoints_predicted[i],
                        max_distance
                        )
                    # check if the trajectory finished or the vector is invalid
                    if not (st[i] and partial_filter):
                        final_filter = self._final_filtering(
                            keypoints_start[i],
                            keypoints_current[i]
                            )
                        # check if it is a valid trajectory
                        if final_filter:
                            velocity_i = _get_velocity(
                                keypoints_start[i],
                                keypoints_current[i],
                                self._pixel_to_real / self._resolution,
                                loader.index - time[i],
                                loader.fps
                                )
                            angle_i = get_angle(
                                keypoints_start[i],
                                keypoints_current[i]
                                )
                            module_start = int(keypoints_start[i].pt[0] /
                                    self._step)
                            module_current = int(keypoints_start[i].pt[0] /
                                    self._step)
                            if module_start == module_current:
                                subregion_velocity[module_start].append(velocity_i)
                                subregion_trajectories[module_start] += 1

                            # update storage
                            pos = i
                            j = loader.index - 1
                            while j >= time[i]:
                                valid[j][pos] = True
                                velocity_mem[j][pos] = velocity_i
                                pos = path[j][pos]
                                j-=1

                            velocity[loader.index].append(velocity_i)
                            angle[loader.index].append(angle_i)
                            distance[loader.index].append(
                                velocity_i * (loader.index - time[i]) / loader.fps)

                        continue

                    # Add new displacement vector
                    keypoints_current[k] = keypoints_current[i]
                    keypoints_start[k] = keypoints_start[i]
                    keypoints_predicted[k] = keypoints_predicted[i]
                    path[loader.index].append(i)
                    velocity_mem[loader.index].append(0)
                    valid[loader.index].append(False)
                    time[k] = time[i]
                    k += 1

                # Only keep until the kth keypoint in order to filter invalid
                # vectors
                keypoints_current = keypoints_current[:k]
                keypoints_start = keypoints_start[:k]
                keypoints_predicted = keypoints_predicted[:k]
                time = time[:k]

            print('number of trajectories:', len(keypoints_current))

            if show_video:
                if previous_frame is not None:
                    color_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2RGB)
                    output = draw_vectors(
                        color_frame,
                        keypoints_predicted,
                        keypoints_current,
                        masks
                        )
                    cv2.imshow("sparse optical flow", output)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            previous_frame = current_frame.copy()
            keypoints_mem_current.append(keypoints_current)
            keypoints_mem_predicted.append(keypoints_predicted)

            # TODO: I guess the swap is not needed such as in the next iteration
            # the keypoints_predicted will be cleaned
            if len(keypoints_predicted) != 0:
                keypoints_predicted, keypoints_current = keypoints_current, keypoints_predicted

        loader.end()
        cv2.destroyAllWindows()
        avg, max_, min_, std_dev, count = compute_stats(velocity)

        print('avg:', avg)
        print('max:', max_)
        print('min:', min_)
        print('std_dev:', std_dev)
        print('count:', count)


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
        # TODO: Perhaps there is a more efficient way to transform this to a
        # numpy array
        new_pt = np.array([int(new.pt[0]), int(new.pt[1])])
        old_pt = np.array([int(old.pt[0]), int(old.pt[1])])
        mask = cv2.line(mask, new_pt.ravel(), old_pt.ravel(), color, thick)

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



def main(config_path: str, video_identifier: str, show_video=True):
    '''Basic example of OTV'''
    loader = get_loader(config_path, video_identifier)
    formatter = Formatter(config_path, video_identifier)
    loader.has_images()
    image = loader.read()
    prev_gray = formatter.apply_distortion_correction(image)
    prev_gray = formatter.apply_roi_extraction(prev_gray)
    otv = OTV(config_path, video_identifier, prev_gray)
    otv.run(loader, formatter, show_video)

if __name__ == "__main__":
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
        '-v',
        '--video',
        action='store_true',
        help='Play video while processing')
    args = parser.parse_args()
    CONFIG_PATH = f'{args.path}/{args.statio_name}.json'
    main(config_path=CONFIG_PATH,
         video_identifier=args.video_identifier,
         show_video=args.video
         )
