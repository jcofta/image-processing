import numpy as np
import cv2

img = cv2.imread("img_ok/probe_cam7.JPG", cv2.IMREAD_COLOR)
cv2.waitKey(0)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 50, 102])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)

kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(mask, kernel, iterations=0)
kernel = np.ones((8, 8), np.uint8)
erosion = cv2.erode(dilation, kernel, iterations=3)
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
    cv2.drawContours(img_copy, [c], -1, (0, 255, 0), 2)
    cv2.circle(img_copy, (cX, cY), 7, (255, 255, 255), -1)

# cv2.drawContours(img_copy, contours, -1, (0,255,0), 3)
# output = cv2.bitwise_and(hsv, hsv, mask=mask)

cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
cv2.imshow('img2', np.hstack([mask, dilation, erosion]))
cv2.imshow('img3', img_copy)
cv2.waitKey(0)