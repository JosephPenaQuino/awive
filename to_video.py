#!/home/joseph/anaconda3/envs/imageProcessing/bin/python3
import numpy as np
import cv2 as cv
# Define the codec and create VideoWriter object
xd = (421, 310)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, xd)
for i in range(100):
    frame = np.load(f'images/im_{i:04d}.npy')
    frame = cv.resize(frame, xd, interpolation = cv.INTER_AREA)
    frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    # write the flipped frame
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
out.release()
cv.destroyAllWindows()
