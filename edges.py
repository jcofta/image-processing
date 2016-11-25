import numpy as np
import cv2
import csv
import math
from skimage import img_as_float

show_images=True

input_img = "probe_cam1"
img = cv2.imread('img_fin/'+ input_img +'.JPG')

cv2.namedWindow('INPUT', cv2.WINDOW_NORMAL)
cv2.imshow('INPUT', img)
cv2.waitKey(0)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def scan_ymax(image):
    height = np.size(image, 0)
    width = np.size(image, 1)

    for y in range(0, height, 1):
        for x in range(0, width, 1):
            if (image.item(y, x) == 255):
                print("Y ", y, "X", x)
                return y,x

def scan_ymin(image):
    height = np.size(image, 0)
    width = np.size(image, 1)

    for y in range(height-1, 0, -1):
        for x in range(width -1, 0, -1):
            if (image.item(y, x) == 255):
                print("Y ", y, "X", x)
                return y,x

def scan_xmax(image):
    height = np.size(image, 0)
    width = np.size(image, 1)

    for x in range(width-1, 0, -1):
        for y in range(0,height, 1):
            if (image.item(y, x) == 255):
                print("Y ", y, "X", x)
                return y,x

def scan_xmin(image):
    height = np.size(image, 0)
    width = np.size(image, 1)

    for x in range( 0,width,1):
        for y in range(height-1, 0, -1):
            if (image.item(y, x) == 255):
                print("Y ", y, "X", x)
                return y,x

#filters on input image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = adjust_gamma(gray, 1.2)
thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]
thresh = cv2.erode(thresh, None, iterations=1)
thresh = cv2.dilate(thresh, None, iterations=5)

#finding corners of the image
y1,x1 = scan_ymax(thresh)
y2,x2 = scan_xmax(thresh)
y3,x3 = scan_ymin(thresh)
y4,x4 = scan_xmin(thresh)

#show threshholded image
if(show_images):
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.imshow('img2', thresh)
    cv2.waitKey(0)


# Destination image size
size = (2048, 1024, 3)

im_dst = np.zeros(size, np.uint8)

pts_dst = np.array(
    [
        [0, 0],
        [size[0] - 1, 0],
        [size[0] - 1, size[1] - 1],
        [0, size[1] - 1]
    ], dtype=float
)

# Calculate the homography
dist_ymax_xmin = math.sqrt( (x4 - x1)**2 + (y4 - y1)**2 )
dist_ymax_xmax  = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

# Find homography depending on rotation of the input image
if(dist_ymax_xmin > dist_ymax_xmax):
	h, status = cv2.findHomography(np.array([[x4,y4],[x1,y1],[x2,y2],[x3,y3]]), pts_dst)
else:
	h, status = cv2.findHomography(np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]), pts_dst)

# Load input image again
img2 = cv2.imread('img_fin/'+ input_img +'.JPG')

# Warp source image to destination
im_dst = cv2.warpPerspective(img2, h, size[0:2])

high = np.size(im_dst, 0)
wide = np.size(im_dst, 1)
print("h = ",high,"w =",wide)

# Save the transformed image
cv2.imwrite('img_fin/' + input_img + '_transformed.JPG',im_dst)

#load template image
img_template = cv2.imread('img_fin/diagram_left_transformed.JPG', 1)

#convert to grayscale
imgray = cv2.cvtColor(img_template,cv2.COLOR_BGR2GRAY)

#find contours
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#draw the contours on the homografed image
for c in contours:
    perimeter = cv2.arcLength(c,True)
    if (perimeter<2000):
        cv2.drawContours(im_dst, [c], -1, (0, 0, 0), 6)

#save image with thick egdes
cv2.imwrite('img_fin/'+ input_img+'_ready.JPG',im_dst)

# Show output
if(show_images):
    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.imshow("Image", im_dst)
    cv2.waitKey(0)

hsv = cv2.cvtColor(im_dst, cv2.COLOR_BGR2HSV)

low_red1 = np.array([0, 50, 102])
up_red1 = np.array([10, 255, 255])

low_red2 = np.array([170,50,50])
up_red2 = np.array([179,255,255])

low_blue = np.array([80, 10, 50])
up_blue = np.array([130, 255, 255])

mask_files = {}
for id in range(1,7,1):
	for lr in ['l', 'r']:
		for ud in ['u', 'd']:
			for side in range(0,5,1):
				filename = str(id) + lr + ud + str(side)
				mask_img = cv2.imread('img_fin/masks/' + filename + '.JPG', 0) #gray
				mask_files[filename] = mask_img.copy()



#match single position with mask
def match_mask(pos):
	#print(pos)
	import time
	for id in range(1,7,1):
		for lr in ['l', 'r']:
			for ud in ['u', 'd']:
				for side in range(0,5,1):
					filename = str(id) + lr + ud + str(side)
					#print(filename)
					# time.sleep(0.1)
					mask_img = mask_files[filename]
					if mask_img.item( pos[1], pos[0]) == 255:
						#print("RETURN")
						return filename
	return None

def match_moments_with_masks(teeth_state, moment_list, value):
	for moment in moment_list:
		res = match_mask(moment)
		if res:
			teeth_state[res] = value
	return teeth_state

def filter_color(lower1,upper1, lower2, upper2):
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask=mask1+mask2
    cv2.namedWindow('color_mask', cv2.WINDOW_NORMAL)
    cv2.imshow('color_mask', mask)
    cv2.waitKey(0)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=2)
    im2, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = im_dst
    moments_list = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            M["m00"] = 1
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(img_copy, (cX, cY), 4, (0, 255, 0), -1)
        moments_list.append([cX, cY])

    cv2.namedWindow('end', cv2.WINDOW_NORMAL)
    cv2.imshow('end', img_copy)
    cv2.waitKey(0)
    return moments_list

teeth_state_glob = {}

red_moments = filter_color(low_red1,up_red1, low_red2, up_red2)
match_moments_with_masks(teeth_state_glob, red_moments, 'prochnica')
blue_moments = filter_color(low_blue,up_blue, low_blue, up_blue)
match_moments_with_masks(teeth_state_glob, blue_moments, 'wypelnienie')

for keys,values in teeth_state_glob.items():
    print(keys + " : " + values)

with open('dict.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, teeth_state_glob.keys())
    w.writeheader()
    w.writerow(teeth_state_glob)
