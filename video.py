'''Play  a video'''
import argparse

import cv2

from loader import get_loader
from loader import Formatter


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'
RESIZE_RATIO = 5


def play(loader: Formatter):
    '''Plays a video'''

    while loader.has_images():
        im = loader.read()
        lil_im = cv2.resize(im, (1000, 1000))
        cv2.imshow('Video', lil_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print ('Finished by key \'q\'')
            break
    cv2.destroyAllWindows()


def main(config_path: str, video_identifier: str):
    '''Read configurations and play video'''
    loader = get_loader(config_path, video_identifier)
    play(loader)


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
