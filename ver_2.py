import cv2
import numpy as np

img = cv2.imread('img_ok/probe_cam4.JPG')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
boundaries = [[340,55, 255], [20, 55, 255]]

#edges=cv2.Canny(img,100,200)


lower = np.array(boundaries[0], dtype='uint8')
upper = np.array(boundaries[1], dtype='uint8')

mask = cv2.inRange(img, lower, upper)
output = cv2.bitwise_and(img, img, mask = mask)

cv2.namedWindow('images', cv2.WINDOW_NORMAL)
cv2.imshow("images", mask)
cv2.waitKey(0)


#cv2.imshow('probe1',red_filter)
#cv2.waitKey(0)
#cv2.destroyAllWindows()





