import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

imagePath = '/home/joseph/Documents/Thesis/Dataset/ssivDataset/video.mp4'
imageDataset = '/home/joseph/Documents/Thesis/Dataset/ssivDataset/images'

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

def main():
    cap = cv.VideoCapture(imagePath)
    ret, frame = cap.read()
    width = frame.shape[0]
    height = frame.shape[1]
    w1 = int(100+width/2)
    w2 = int(2*width/2)
    h1 = int(height/2)
    h2 = int(2*height/2)
    # h1 = 800
    # h2 = 2432
    # w1 = 646
    # w2 = 1423
    h1 = 300
    h2 = 700
    w1 = 360
    w2 = 660

    w = w2 - w1
    h = h2 - h1

    frameNumber = 1

    sti = [np.empty((1, h), dtype=np.uint8) for i in range(frameNumber)]
    # sti = np.zeros((1, 853))
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

    cap = cv.VideoCapture(imagePath)
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
    main()
