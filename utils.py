import cv2
import numpy as np


def dilation_erosion(edges, kernel_size=5, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=iterations)
    eroded = cv2.erode(dilated, kernel, iterations=iterations)
    return eroded
