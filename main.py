import numpy as np
import cv2

img = cv2.imread("img/probe_cam5.JPG", cv2.IMREAD_COLOR)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

boundaries = ([17, 15, 100], [50, 56, 200])

lower = np.array([0, 50, 50], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

mask = cv2.inRange(img, lower, upper)
output = cv2.bitwise_and(img, img, mask=mask)

img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
# show the images
cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow('img2', output)
cv2.waitKey(0)
