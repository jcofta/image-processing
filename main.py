import numpy as np
import cv2

img = cv2.imread("img_ok/probe_cam1.JPG", cv2.IMREAD_COLOR)
cv2.waitKey(0)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)

kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(mask, kernel, iterations=0)
kernel = np.ones((10,10), np.uint8)
erosion = cv2.erode(dilation, kernel, iterations=3)
# output = cv2.bitwise_and(hsv, hsv, mask=mask)

cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow('img2', np.hstack([mask, dilation, erosion]))
# cv2.imshow('img2', np.hstack([img, output]))
cv2.waitKey(0)
