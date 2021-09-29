import cv2 as cv

class ImageLoader:
    def __init__(self, imageDataset, prefix, digits, offset=0):
        self._image_dataset = imageDataset
        self._offset = offset
        self._prefix = prefix
        self._digits = digits

    def path(self, i):
        i += self._offset
        if self._digits == 5:
            return f'{self._image_dataset}/{self._prefix}{i:05}.jpg'
        elif self._digits == 3:
            return f'{self._image_dataset}/{self._prefix}{i:03}.jpg'
        else:
            return f'{self._image_dataset}/{self._prefix}{i:04}.jpg'

    def load(self, index):
        return cv.imread(self.path(index), cv.IMREAD_GRAYSCALE)
