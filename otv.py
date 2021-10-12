import cv2 as cv
import numpy as np

class OTV():
    def __init__(self, prev_gray):
        self.feature_params = dict(maxCorners = 300,
                                   qualityLevel = 0.2,
                                   minDistance = 4,
                                   blockSize = 2)
        self.lk_params = dict(winSize = (5, 5),
                              maxLevel = 2, 
                              criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 
                                          10,
                                          0.03))
        self.prev_gray = prev_gray
        prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **self.feature_params)
    def calc(self, gray):
        self.prev = cv.goodFeaturesToTrack(self.prev_gray, mask = None, **self.feature_params)
        next, status, error = cv.calcOpticalFlowPyrLK(self.prev_gray, 
                                                      gray,
                                                      self.prev,
                                                      None,
                                                      **self.lk_params)
        good_old = self.prev[status == 1].astype(int)
        good_new = next[status == 1].astype(int)
        self.prev_gray = gray.copy()
        self.prev = good_new.reshape(-1, 1, 2)
        return good_old, good_new


def main():
    cap = cv.VideoCapture("/home/joseph/Desktop/video_1.mp4")
    color = (0, 255, 0)
    ret, first_frame = cap.read()
    h1 = 1115
    h2 = 1800
    w1 = 330
    w2 = 1050
    first_frame = first_frame[w1:w2, h1:h2]
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    otv = OTV(prev_gray)

    while(cap.isOpened()):
        # Get image
        ret, frame = cap.read()
        frame = frame[w1:w2, h1:h2]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        good_old, good_new = otv.calc(gray)

        # Draw vectors
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color, 2)
            frame = cv.circle(frame, (a, b), 3, color, -1)
        output = cv.add(frame, mask)

        # Plot frame
        cv.imshow("sparse optical flow", output)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
