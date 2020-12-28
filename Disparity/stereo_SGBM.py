# disparity map calculation with semi-global block matching
import numpy as np
import cv2 as cv
import timeit

print("Loading Images")

#path to left image
path1 = "images/imL.png"

#path to right image
path2 = r"images/imR.png"
imgL = cv.imread(path1,cv.IMREAD_GRAYSCALE) 
imgR = cv.imread(path2,cv.IMREAD_GRAYSCALE) 
window_size = 3
min_disp = 0
num_disp = 64 - min_disp
stereo = cv.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 7,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
         
    )
print("computing disparity")
start= timeit.default_timer()
disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
end = timeit.default_timer()
time_taken = end - start
print(f"time taken {time_taken} usec")
cv.imshow('disparity', (disp-min_disp)/num_disp)
cv.waitKey()

