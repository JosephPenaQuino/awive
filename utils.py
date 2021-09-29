import cv2 as cv

class ImageLoader:
    def __init__(self, imageDataset):
        self._imageDataset = imageDataset

    def path(self, i):
        i += 200
        return f'{self.imageDataset}/out-{i:03}.jpg'

    def load(self, index):
        return cv.imread(self.path(index), cv.IMREAD_GRAYSCALE)
