import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

imageDataset = '/home/joseph/Documents/Thesis/Dataset/ssivDataset/images'
imageLength = 372


def image_path(i):
    i += 200
    return f'{imageDataset}/out-{i:03}.jpg'


def load_image(index):
    return cv.imread(image_path(index), cv.IMREAD_GRAYSCALE)


def image_mean():
    imshape = load_image(0).shape
    image_acumulator = np.zeros(imshape, dtype=np.int64)
    for i in range(imageLength):
        image_acumulator += load_image(i)
    return (image_acumulator / imageLength).astype(np.uint8)


def main():
    imMean = image_mean()
    cv.imshow('image mean', imMean)
    # for i in range(3):
    #     cv.imshow(f'frame: {i:03}', load_image(i) - imMean)
    im = load_image(0)
    for i in range(10, 100, 20):
        s = (imMean*(i/100)).astype(np.uint8)
        cv.imshow(f'k = {i:02}%', im-s)

    cv.imshow('original image', im)
    cv.waitKey(0) 
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
