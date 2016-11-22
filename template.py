import numpy as np
import cv2
import imutils
import math
from skimage import img_as_float

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

template = cv2.imread('img_ok2/tooth5.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = auto_canny(template)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)

image = cv2.imread('img_ok2/diagram_new.jpg')
#image = cv2.imread('img_ok2/probe_cam1_transformed.JPG')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
cv2.imshow('Input Image', image)

##PREPARE EDGES
#gaussian filter
#image_blure = cv2.GaussianBlur(image, (5, 5), 0)


###################### PROBY WYRZUCENIA KOLORÓW
#hsv_blure = cv2.cvtColor(image_blure, cv2.COLOR_BGR2HSV)
#low_red1 = np.array([0, 50, 102])
#up_red1 = np.array([10, 255, 255])
#low_red2 = np.array([170,50,50])
#up_red2 = np.array([179,255,255])

#mask1 = cv2.inRange(hsv_blure, low_red1, up_red1)
#mask2 = cv2.inRange(hsv_blure, low_red2, up_red2)
#mask=mask1+mask2
#print(mask)

#height = np.size(hsv_blure, 0)
#width = np.size(hsv_blure, 1)

#for x in range( 0,width,1):
#	for y in range(0, height, 1):
#		if (mask.item(y,x)==255):
#			print("Y= ",y, "X=",x)
#			image.item(y,x)=(0,0,0)


#cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
#cv2.imshow('mask', mask)

#masked_img = cv2.bitwise_and(image,image,mask = mask)

#cv2.namedWindow('masked image', cv2.WINDOW_NORMAL)
#cv2.imshow('masked image', masked_img)
#cv2.namedWindow('HSV Blurred',cv2.WINDOW_NORMAL)
#cv2.imshow('HSV Blurred', hsv_blure)
###############################


#dst = np.zeros(shape=(tW,tH))
#dst = cv2.normalize(image_blure, dst, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#cv2.namedWindow('Norm hist Image',cv2.WINDOW_NORMAL)
#cv2.imshow('Norm hist Image', dst)


##
image_gamma = adjust_gamma(image, 1.2)

thresh = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY_INV)[1]
# thresh = cv2.erode(thresh, None, iterations=1)
# thresh = cv2.dilate(thresh, None, iterations=1)

cv2.namedWindow('Thresh Image',cv2.WINDOW_NORMAL)
cv2.imshow('Thresh Image', thresh)

#image_canny = auto_canny(thresh)

#cv2.namedWindow('Canny',cv2.WINDOW_NORMAL)
#cv2.imshow('Canny', image_canny)

# gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
#gray = image_canny
gray=thresh

found = None
i=0

# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.44
# loc = np.where( res >= threshold)
# for pt in zip(*loc[::-1]):
# cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

for scale in np.linspace(1.0, 0.8, 1):


# resize the image according to the scale, and keep track
# of the ratio of the resizing
	resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
	r = gray.shape[1] / float(resized.shape[1])
		# if the resized image is smaller than the template, then break
	# from the loop
	if resized.shape[0] < tH or resized.shape[1] < tW:
		break
	# detect edges in the resized, grayscale image and apply template
	# matching to find the template in the image
	edged = auto_canny(resized)
	result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
	loc = np.where( result >= threshold)

	(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

	print(scale, maxVal, maxLoc, loc)
	if found is None or maxVal > found[0]:
		found = (maxVal, maxLoc, r)
	i=i+1
	# print(i, maxVal, maxLoc, r)
# unpack the bookkeeping varaible and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
# print(result)
for pt in zip(*loc[::-1]):
	cv2.rectangle(image, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)

(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

# draw a bounding box around the detected result and display the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.waitKey(0)
