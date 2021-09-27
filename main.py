import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

show_histogram = False

# while True:
cap = cv.VideoCapture("video.mp4")
ret, frame = cap.read()
width = frame.shape[0]
height = frame.shape[1]

# Show histogram
if show_histogram:
    imm = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    plt.hist(imm.ravel(),256,[0,256])
    plt.show()
    exit()

cont = True
alpha = 0.8
beta = 50

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

while cont:
    cap = cv.VideoCapture("video.mp4")
    while cap.isOpened():
        # Read image
        ret, frame = cap.read()
        if not ret:
            break
        imm = frame[int(100+width/3):int(2*width/3), int(height/3):int(2*height/3)]
        imm = cv.cvtColor(imm, cv.COLOR_BGR2GRAY)

        # imm = cv.equalizeHist(imm)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        imm = clahe.apply(imm)
        imm = (255.0*(((imm - 80).astype("uint8")) / 120)).astype("uint8")
        imm = adjust_gamma(imm, 0.9)
        # imm = cv.convertScaleAbs(imm, alpha=alpha, beta=beta)

        cv.imshow("my video", imm)

        if cv.waitKey(50) & 0xFF == ord('q'):
            cont = False
            break
cap.release()
cv.destroyAllWindows()
