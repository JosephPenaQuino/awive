from utils import VideoLoader
from utils import Formatter
import sys
import json
import cv2 as cv


def main(option):
    # Read configuration
    conf = json.load(open('conf/sti.json'))[option]
    loader = VideoLoader(conf['video_path'], conf['offset'])
    im = loader.read()
    formatter = Formatter(im.shape,
                         conf['rotate_image'],
                         1.0,
                         conf['roi']['w1'],
                         conf['roi']['w2'],
                         conf['roi']['h1'],
                         conf['roi']['h2']
                         )
    im = formatter.apply(loader.read())
    cv.imwrite("tmp.png", im)
    cv.imshow("frame", im)
    cv.waitKey(0) 
    cv.destroyAllWindows()


if __name__ == "__main__":
    option = 'd5'
    if len(sys.argv) > 1:
        option = sys.argv[1]
    main(option)
