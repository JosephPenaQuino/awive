'''Optical Tracking Image Velocimetry'''

import argparse
import numpy as np
import cv2
from correct_image import Formatter
from loader import get_loader
# from sti import applyMean


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'


class OTV():
    '''Optical Tracking Image Velocimetry'''
    def __init__(self, prev_gray):
        self.feature_params = dict(maxCorners=300,
                                   qualityLevel=0.2,
                                   minDistance=4,
                                   blockSize=2)
        self.lk_params = dict(
            winSize=(5, 5),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03))
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
    masks = []
    otv = OTV(prev_gray)

    while loader.has_images():
        color_frame = loader.read()
        frame = formatter.apply_roi_extraction(color_frame)
        good_old, good_new = otv.calc(frame)
        output = draw_vectors(frame, good_new, good_old, masks)

        # Plot frame
        cv2.imshow("sparse optical flow", output)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    loader.end()
    cv2.destroyAllWindows()

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
