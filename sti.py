import cv2 as cv
import sys
from matplotlib import pyplot as plt
import numpy as np
import json
from utils import ImageLoader


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
    video_path = conf['video_path']

    # Read video
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    width = frame.shape[0]
    height = frame.shape[1]
    h1 = conf['roi']['h1']
    h2 = conf['roi']['h2']
    w1 = conf['roi']['w1']
    w2 = conf['roi']['w2']
    w = w2 - w1
    h = h2 - h1

    frameNumber = 1

    sti = [np.empty((1, h), dtype=np.uint8) for i in range(frameNumber)]
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))


    for i in range(55):
        ret, frame = cap.read()

    # Get average 
    imMean =  np.zeros((w, h), dtype=np.int64)
    cnt = 0
    while cap.isOpened():
        ret, imm = cap.read()
        if not ret:
            break
        M = cv.getRotationMatrix2D((width//2, height//2), -25, 1.0)
        imm = cv.warpAffine(imm, M, (width, height))
        imm = imm[w1:w2, h1:h2]
        imm = cv.cvtColor(imm, cv.COLOR_BGR2GRAY)
        imm = imm.astype(int)
        imMean += imm
        cnt += 1

    print('cnt:', cnt)
    print('imMean:', np.max(imMean), np.min(imMean))
    imMean = (imMean / cnt) * 1
    imMean = imMean.astype(np.uint8)
    print('imMean:', np.max(imMean), np.min(imMean))
    cv.imshow('image mean', imMean)

    cap = cv.VideoCapture(video_path)
    for i in range(55):
        ret, frame = cap.read()


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        imm = frame
        M = cv.getRotationMatrix2D((width//2, height//2), -25, 1.0)
        imm = cv.warpAffine(imm, M, (width, height))
        imm = imm[w1:w2, h1:h2]
        imm = cv.cvtColor(imm, cv.COLOR_BGR2GRAY)
        # imm -= imMean
        # imm = clahe.apply(imm)
        # imm = (255.0*(((imm - 80).astype("uint8")) / 120)).astype("uint8")
        # imm = adjust_gamma(imm, 0.9)
        # np.save("tmp2.npy", imm)
        for i in range(frameNumber):
            offset = i*10
            ref = w//2 + offset
            row = imm[ref:ref+ 1, :]
            sti[i] = np.vstack([sti[i], row])
    
    for i in range(frameNumber):
        cv.imshow(f'sti_{i:02}', sti[i])
    cv.imshow("frame", imm)
    cv.waitKey(0) 
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    option = 'd1'
    if len(sys.argv) > 1:
        option = sys.argv[1]
    main(option)
