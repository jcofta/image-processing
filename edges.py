import numpy as np
import cv2

img = cv2.imread("img_ok/probe_cam1.JPG")
cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]
thresh = cv2.dilate(thresh, None, iterations=3)
#thresh = cv2.erode(thresh, None, iterations=6)

cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow('img2', thresh)
cv2.waitKey(0)

img_copy=img
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
	cv2.drawContours(img_copy, [c], -1, (0, 255, 0), -1)
	
cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
cv2.imshow('img3', img_copy)
cv2.waitKey(0)


