import cv2
import numpy as np


def tenengrad_rgb(image):
    b, g, r = cv2.split(image)

    grad_x_b = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_b = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)

    grad_x_g = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_g = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)

    grad_x_r = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_r = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=3)

    magnitude_squared = (grad_x_b**2 + grad_y_b**2 +
                         grad_x_g**2 + grad_y_g**2 +
                         grad_x_r**2 + grad_y_r**2)

    sharpness = np.mean(magnitude_squared)

    return sharpness


if __name__ == '__main__':
    image = cv2.imread('temp-images/0000.jpg')
    sharpness = tenengrad_rgb(image)
    print(f'Sharpness: {sharpness}')
