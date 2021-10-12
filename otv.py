import cv2 as cv
import sys
import json
import numpy as np
from utils import Formatter
from utils import VideoLoader
from sti import scaleImage, applyMean, adjust_gamma

class OTV():
    def __init__(self, prev_gray):
        self.feature_params = dict(maxCorners = 300,
                                   qualityLevel = 0.2,
                                   minDistance = 4,
                                   blockSize = 2)
        self.lk_params = dict(winSize = (5, 5),
                              maxLevel = 2, 
                              criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 
                                          10,
                                          0.03))
        self.prev_gray = prev_gray
        prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **self.feature_params)

    def calc(self, gray):
        self.prev = cv.goodFeaturesToTrack(self.prev_gray, mask = None, **self.feature_params)
        next, status, error = cv.calcOpticalFlowPyrLK(self.prev_gray, 
                                                      gray,
                                                      self.prev,
                                                      None,
                                                      **self.lk_params)
        good_old = self.prev[status == 1].astype(int)
        good_new = next[status == 1].astype(int)
        self.prev_gray = gray.copy()
        self.prev = good_new.reshape(-1, 1, 2)
        return good_old, good_new


def main(option):
    # Read configuration
    conf = json.load(open('conf/sti.json'))[option]
    loader = VideoLoader(conf['video_path'], conf['offset'])
    h1 = conf['roi']['h1']
    h2 = conf['roi']['h2']
    w1 = conf['roi']['w1']
    w2 = conf['roi']['w2']
    w = w2 - w1
    h = h2 - h1
    # Set formatter
    loader.has_images()
    im = loader.read()

    formatter = Formatter(im.shape,
                         0,
                         1.0,
                         conf['roi']['w1'],
                         conf['roi']['w2'],
                         conf['roi']['h1'],
                         conf['roi']['h2'])


    # Get average 
    imMean =  np.zeros((w, h), dtype=np.int64)
    cnt = 0
    while loader.has_images():
        im = formatter.apply(loader.read()).astype(int)
        imMean += im
        cnt += 1
    imMean = (imMean / cnt) * 1
    imMean = imMean.astype(np.uint8)
    # imMean = adjust_gamma(imMean, 0.8)

    cv.imshow('image mean', imMean)
    cv.waitKey(0) 
    cv.destroyAllWindows()
    loader.finish()

    # apply otv
    loader = VideoLoader(conf['video_path'], conf['offset'])
    loader.has_images()
    color = (0, 255, 0)
    colorframe = loader.read()
    prev_gray = formatter.apply(colorframe)
    prev_gray = applyMean(prev_gray, imMean, 0.5)
    colorframe = colorframe[w1:w2, h1:h2]
    cv.imshow('prev gray', prev_gray)
    cv.waitKey(0) 
    cv.destroyAllWindows()


    mask = np.zeros_like(colorframe)
    otv = OTV(prev_gray)

    while loader.has_images():
        # Get image
        colorframe = loader.read()
        frame = formatter.apply(colorframe)
        frame = applyMean(frame, imMean, 0.5)
        colorframe = colorframe[w1:w2, h1:h2]
        good_old, good_new = otv.calc(frame)

        # Draw vectors
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color, 2)
            frame = cv.circle(frame, (a, b), 3, color, -1)
        output = cv.add(colorframe, mask)

        # Plot frame
        cv.imshow("sparse optical flow", output)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    loader.finish()
    cv.destroyAllWindows()

if __name__ == '__main__':
    option = 'd2'
    if len(sys.argv) > 1:
        option = sys.argv[1]
    main(option)
