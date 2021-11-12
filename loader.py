import cv2 as cv
import abc


class Loader(metaclass=abc.ABCMeta):
    def __init__(self, offset):
        self._offset = offset
        pass

    @abc.abstractmethod
    def has_images(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass

class ImageLoader:
    def __init__(self, imageDataset, prefix, digits, offset=0):
        super().__init__(offset)
        self._image_dataset = imageDataset
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

class VideoLoader(Loader):
    def __init__(self, video_path, offset=0):
        super().__init__(offset)
        self._video_path = video_path
        self._cap = cv.VideoCapture(self._video_path)
        self._image = None # Current image
        self._image_read = False # Check if the current images was read

        # Skip offset
        for _ in range(self._offset+1):
            if self.has_images():
                self.read()

    def has_images(self):
        if not self._cap.isOpened():
            return False
        ret, self._image = self._cap.read()
        self._image_read = False
        return ret

    def read(self):
        if self._image_read:
            _, self._image = self._cap.read()
        self._image_read = True
        return self._image

    def finish(self):
        self._cap.release()

class Formatter:
    def __init__(self, shape, grades, a, w1, w2, h1, h2, gray=True):
        self._width = shape[0]
        self._height = shape[1]
        self._grades = grades
        self._a = a
        self._M = cv.getRotationMatrix2D((self._width//2, self._height//2),
                                         grades,
                                         a)
        self._w1 = w1
        self._w2 = w2
        self._h1 = h1
        self._h2 = h2
        self._gray = gray

    def apply(self, image):
        # Rotate image
        if self._grades != 0:
            image = cv.warpAffine(image, self._M, (self._width, self._height))
        # Crop image
        image = image[self._w1:self._w2, self._h1:self._h2]
        # To gray
        if self._gray:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return image
