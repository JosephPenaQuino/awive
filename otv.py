'''Optical Tracking Image Velocimetry'''

import argparse
import json
import numpy as np
import cv2
from correct_image import Formatter
from loader import (get_loader, Loader)
# from sti import applyMean


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'


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
        winsize = config['lk']['winsize']

        self.lk_params = {
            'winSize': (winsize, winsize),
            'maxLevel': config['lk']['maxlevel'],
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         10,
                         0.03)
            }
        self.prev_gray = prev_gray
        self.prev = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=None,
            **self.feature_params)

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
        max_features = 3000  # TODO: this is an example
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
                    if len(keypoints_current) < max_features:
                        keypoints_current.append(keypoint)
                        keypoints_start.append(keypoint)
                        time.append(loader.index)
                        valid[loader.index].append(False)
                        velocity_mem[loader.index].append(0)
                        path[loader.index].append(i)
            else:
                for i, keypoint in enumerate(reversed(keypoints)):
                    if len(keypoints_current) < max_features:
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
                pts2 = pts2.astype(int)
                pts1 = pts1.astype(int)
                if pts2 is not None:
                    good_new = pts2[(st==1).T[0]]
                    good_old = pts1[(st==1).T[0]]
                # keypoints_predicted.clear()
                # for pt2 in pts2:
                #     keypoints_predicted.append(cv2.KeyPoint(pt2, 1.0))

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
