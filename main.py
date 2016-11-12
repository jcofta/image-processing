import numpy as np
import cv2

img = cv2.imread("img_ok/probe_cam3_transformed.JPG", 1)
cv2.waitKey(0)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low_red = np.array([0, 50, 102])
up_red = np.array([10, 255, 255])

low_blue = np.array([90, 50, 102])
up_blue = np.array([120, 255, 255])


def filter_color(lower,upper):
	mask = cv2.inRange(hsv, lower, upper)

	kernel = np.ones((3, 3), np.uint8)
	dilation = cv2.dilate(mask, kernel, iterations=0)
	kernel = np.ones((8, 8), np.uint8)
	erosion = cv2.erode(dilation, kernel, iterations=2)
	im2, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	img_copy = img
	# loop over the contours
	for c in contours:
    	# compute the center of the contour
    		M = cv2.moments(c)
    		if M["m00"] == 0:
        		M["m00"] = 1
    		cX = int(M["m10"] / M["m00"])
    		cY = int(M["m01"] / M["m00"])

    # draw the contour and center of the shape on the image
    		cv2.drawContours(img_copy, [c], -1, (0, 255, 0), 1)
    		cv2.circle(img_copy, (cX, cY), 2, (0, 255, 0), -1)
	

	cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
	cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
	cv2.imshow('img2', np.hstack([mask, dilation, erosion]))
	cv2.imshow('img3', img_copy)
	cv2.waitKey(0)

filter_color(low_red,up_red)
#filter_color(low_blue,up_blue)
