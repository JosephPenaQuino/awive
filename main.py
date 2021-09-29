import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

show_histogram = False
imagePath = '/home/joseph/Documents/Thesis/Dataset/ssivDataset/video.mp4'

# while True:
# cap = cv.VideoCapture("video.mp4")
cap = cv.VideoCapture(imagePath)
ret, frame = cap.read()
width = frame.shape[0]
height = frame.shape[1]

# Values for ssiv dataset
h1 = 300
h2 = 700
w1 = 360
w2 = 660

# Index gathered by viewing the image using plotNpy
# h1 = 800
# h2 = 2432
# w1 = 646
# w2 = 1423

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
    cap = cv.VideoCapture("/home/joseph/Documents/Thesis/Dataset/ssivDataset/video.mp4")
    for i in range(55):
        ret, frame = cap.read()

    while cap.isOpened():
        # Read image
        ret, frame = cap.read()
        if not ret:
            break
        # imm = frame[int(100+width/3):int(2*width/3), int(height/3):int(2*height/3)]
        imm = frame
        # imm = frame[w1:w2, h1:h2]
        # imm = frame
        imm = cv.cvtColor(imm, cv.COLOR_BGR2GRAY)
        M = cv.getRotationMatrix2D((width//2, height//2), -25, 1.0)
        imm = cv.warpAffine(imm, M, (width, height))
        imm = imm[w1:w2, h1:h2]
        np.save("tmp.npy", imm)

        # imm = cv.equalizeHist(imm)
        # clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        # imm = clahe.apply(imm)
        # imm = (255.0*(((imm - 80).astype("uint8")) / 120)).astype("uint8")
        # imm = adjust_gamma(imm, 0.9)
        # imm = cv.convertScaleAbs(imm, alpha=alpha, beta=beta)

        cv.imshow("my video", imm)

        if cv.waitKey(50) & 0xFF == ord('q'):
            cont = False
            break
cap.release()
cv.destroyAllWindows()
