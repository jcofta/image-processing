import numpy as np
import cv2

img = cv2.imread("img_ok2/probe_cam1_ready.JPG", 1)
cv2.waitKey(0)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low_red1 = np.array([0, 50, 102])
up_red1 = np.array([10, 255, 255])

low_red2 = np.array([170,50,50])
up_red2 = np.array([179,255,255])


#low_blue = np.array([80, 50, 50])
#up_blue = np.array([130, 255, 255])

def filter_color(lower1,upper1, lower2, upper2):
	
	mask1 = cv2.inRange(hsv, lower1, upper1)
	mask2 = cv2.inRange(hsv, lower2, upper2)
	mask=mask1+mask2
	cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
	cv2.imshow('mask', mask)
	
	
	kernel = np.ones((5, 5), np.uint8)
	erosion = cv2.erode(mask, kernel, iterations=2)
	cv2.namedWindow('erosion', cv2.WINDOW_NORMAL)
	cv2.imshow('erosion', erosion)
	
	#kernel = np.ones((1, 1), np.uint8)
	#dilation = cv2.dilate(mask, kernel, iterations=0)
	#cv2.namedWindow('dilation', cv2.WINDOW_NORMAL)
	#cv2.imshow('dilation', dilation)
	

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
    		#cv2.drawContours(img_copy, [c], -1, (0, 255, 0), 1)
    		cv2.circle(img_copy, (cX, cY), 4, (0, 255, 0), -1)
	

	cv2.namedWindow('end', cv2.WINDOW_NORMAL)
	
	cv2.imshow('end', img_copy)
	cv2.waitKey(0)

filter_color(low_red1,up_red1, low_red2, up_red2)
#filter_color(low_blue,up_blue, low_blue, up_blue)
