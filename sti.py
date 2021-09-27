import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def main():
    cap = cv.VideoCapture("video.mp4")
    ret, frame = cap.read()
    width = frame.shape[0]
    height = frame.shape[1]
    w1 = int(100+width/3)
    w2 = int(2*width/3)
    h1 = int(height/3)
    h2 = int(2*height/3)

    sti = np.empty((1, 853), dtype=np.uint8)
    # sti = np.zeros((1, 853))
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        imm = frame[w1:w2, h1:h2]
        imm = cv.cvtColor(imm, cv.COLOR_BGR2GRAY)
        imm = clahe.apply(imm)
        imm = (255.0*(((imm - 80).astype("uint8")) / 120)).astype("uint8")
        w = w2 - w1
        h = h2-h1
        # M = cv.getRotationMatrix2D((w//2, h//2), -20, 1.0)
        # imm = cv.warpAffine(imm, M, (w, h))
        row = imm[w//2:w//2 + 1, :]
        sti = np.vstack([sti, row])

    cv.imshow("sti", sti)
    cv.imshow("frame", imm)
    cv.waitKey(0) 
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
