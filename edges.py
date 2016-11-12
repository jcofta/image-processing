import numpy as np
import cv2

img = cv2.imread("img_ok/probe_cam3.JPG")
cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


gray = adjust_gamma(gray, 1.2)

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

    for x in range(0, width, 1):
        for y in range(0, height, 1):
            if (image.item(y, x) == 255):
                print("Y ", y, "X", x)
                return y,x

def scan_xmin(image):
    height = np.size(image, 0)
    width = np.size(image, 1)

    for x in range(width-1, 0, -1):
        for y in range(height-1, 0, -1):
            if (image.item(y, x) == 255):
                print("Y ", y, "X", x)
                return y,x

thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]
thresh = cv2.erode(thresh, None, iterations=1)
thresh = cv2.dilate(thresh, None, iterations=5)

y1,x1 = scan_ymax(thresh)
y2,x2 = scan_ymin(thresh)
y3,x3 = scan_xmax(thresh)
y4,x4 = scan_xmin(thresh)

cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow('img2', thresh)
cv2.waitKey(0)

img_copy = img
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.circle(img_copy,(x1,y1), 30, (0,0,255), -1)
cv2.circle(img_copy,(x2,y2), 30, (0,0,255), -1)
cv2.circle(img_copy,(x3,y3), 30, (0,0,255), -1)
cv2.circle(img_copy,(x4,y4), 30, (0,0,255), -1)

for c in contours:
    cv2.drawContours(img_copy, [c], -1, (0, 255, 0), -1)

cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
cv2.imshow('img3', img_copy)

# Destination image
size = (2000, 1000, 3)

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
h, status = cv2.findHomography(np.array([[x3,y3],[x1,y1],[x4,y4],[x2,y2]]), pts_dst)


img2 = cv2.imread("img_ok/probe_cam3.JPG")
# Warp source image to destination
im_dst = cv2.warpPerspective(img2, h, size[0:2])


high = np.size(im_dst, 0)
wide = np.size(im_dst, 1)
print("h = ",high,"w =",wide)

cv2.imwrite("img_ok/probe_cam3_transformed.JPG",im_dst)

# Show output
# cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow("Image", im_dst)


cv2.waitKey(0)
