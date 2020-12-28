import numpy as np
import cv2 as cv
import timeit

print("Loading Images")
path1 = r"C:\Users\tanma\OneDrive\Desktop\internships\ESSI_tech\KIITI Dataset\ALL-2views\Books\view1.png"

#path to right image
path2 = r"C:\Users\tanma\OneDrive\Desktop\internships\ESSI_tech\KIITI Dataset\ALL-2views\Books\view5.png"

img1 = cv.imread(path1,cv.IMREAD_GRAYSCALE) 
img2 = cv.imread(path2,cv.IMREAD_GRAYSCALE) 

print("computing disparity")

start= timeit.default_timer()
stereo = cv.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(img1,img2).astype(np.float32) / 16.0
end = timeit.default_timer()
total_time = end-start
print(f"time taken {total_time} usec")
cv.imshow('disparity',disparity/64)
cv.waitKey()
