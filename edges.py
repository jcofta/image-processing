import numpy as np
import cv2

img = cv2.imread("img_ok/probe_cam1.JPG")
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
        exit = False
        for x in range(0, width, 1):
            if (image.item(y, x) == 255):
                exit = True
                print("Y ", y, "X", x)
                break
        if exit:
            break

def scan_ymin(image):
    height = np.size(image, 0)
    width = np.size(image, 1)

    for y in range(height-1, 0, -1):
        exit = False
        for x in range(width -1, 0, -1):
            if (image.item(y, x) == 255):
                exit = True
                print("Y ", y, "X", x)
                break
        if exit:
            break

def scan_xmax(image):
    height = np.size(image, 0)
    width = np.size(image, 1)

    for x in range(0, width, 1):
        exit = False
        for y in range(0, height, 1):
            if (image.item(y, x) == 255):
                exit = True
                print("Y ", y, "X", x)
                break
        if exit:
            break

def scan_xmin(image):
    height = np.size(image, 0)
    width = np.size(image, 1)

    for x in range(width-1, 0, -1):
        exit = False
        for y in range(height-1, 0, -1):
            if (image.item(y, x) == 255):
                exit = True
                print("Y ", y, "X", x)
                break
        if exit:
            break

thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]
thresh = cv2.erode(thresh, None, iterations=1)
thresh = cv2.dilate(thresh, None, iterations=5)

scan_ymax(thresh)
scan_ymin(thresh)

scan_xmax(thresh)
scan_xmin(thresh)

cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow('img2', thresh)
cv2.waitKey(0)

img_copy = img
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    cv2.drawContours(img_copy, [c], -1, (0, 255, 0), -1)

cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
cv2.imshow('img3', img_copy)
cv2.waitKey(0)
