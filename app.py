'''Main application of Adaptive Water Image Velocimetry Estimator'''

import argparse

import cv2

from loader import get_loader


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'


def main(config_path: str, video_identifier: str):
    '''Use basic loader functions to read an image '''
    loader = get_loader(config_path, video_identifier)
    image = loader.read()
    cv2.imshow('image', image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    loader.end()


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
