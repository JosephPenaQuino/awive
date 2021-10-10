import cv2 as cv
import sys
from matplotlib import pyplot as plt
import numpy as np
import json
from utils import VideoLoader
from utils import Formatter


class STIV():
    def __init__(self, shape, frames_qnt, ref):
        self._frames_qnt = frames_qnt
        self._ref = ref
        self._width = shape[0]
        self._height = shape[1]
        self._stis = [np.empty((1, self._height), dtype=np.uint8) for i in
                range(frames_qnt)]


    def append(self, image):
        for i in range(self._frames_qnt):
            offset = i*10
            ref = self._width//2 + offset
            row = image[ref:ref+ 1, :]
            self._stis[i] = np.vstack([self._stis[i], row])

    @property
    def stis(self, idx):
        return self._stis[idx]

def scaleImage(image):
    image = image.astype('float')
    mmin = np.max(image)
    r = mmin - np.min(image)
    image = 255.0*((image - np.min(image))/r)
    return image.astype('uint8')

def applyMean(im, imMean, ratio):
        tmpMean = imMean * ratio 
        tmpMean = tmpMean.astype('uint8')
        im = im - tmpMean
        return scaleImage(im)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

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
    im = loader.read()
    formatter = Formatter(im.shape,
                         -25,
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
    cv.imshow('image mean', imMean)
    loader.finish()

    # Apply STIV
    loader = VideoLoader(conf['video_path'], conf['offset'])
    frameNumber = 1
    ref = 9
    stiv = STIV((w, h), frameNumber, ref)

    # clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    while loader.has_images():
        im = formatter.apply(loader.read())
        # im = applyMean(im, imMean, 0.5)

        # imm = clahe.apply(imm)
        # imm = (255.0*(((imm - 80).astype("uint8")) / 120)).astype("uint8")
        # imm = adjust_gamma(imm, 0.9)

        stiv.append(im)
    
    for i in range(frameNumber):
        cv.imshow(f'sti_{i:02}', stiv._stis[i])
    cv.imshow("frame", im)
    cv.waitKey(0) 
    cv.destroyAllWindows()
    loader.finish()


if __name__ == '__main__':
    option = 'd0'
    if len(sys.argv) > 1:
        option = sys.argv[1]
    main(option)
