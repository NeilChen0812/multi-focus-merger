import cv2
import numpy as np


def tenengrad_rgb(image):
    if not isinstance(image, np.ndarray):
        raise ValueError('Input must be a numpy array but got {}'.format(
            type(image)))
    if image.ndim != 3:
        raise ValueError('Input must be a color image but got {} image dimension'.format(
            image.ndim))

    # measure the gradient in each channel
    b_channel, g_channel, r_channel = cv2.split(image)

    grad_x_b = cv2.Sobel(b_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_b = cv2.Sobel(b_channel, cv2.CV_64F, 0, 1, ksize=3)

    grad_x_g = cv2.Sobel(g_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_g = cv2.Sobel(g_channel, cv2.CV_64F, 0, 1, ksize=3)

    grad_x_r = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_r = cv2.Sobel(r_channel, cv2.CV_64F, 0, 1, ksize=3)

    # calculate the magnitude squared
    magnitude_squared = (grad_x_b**2 + grad_y_b**2 +
                         grad_x_g**2 + grad_y_g**2 +
                         grad_x_r**2 + grad_y_r**2)

    sharpness = int(np.mean(magnitude_squared))

    return sharpness


if __name__ == '__main__':
    image = cv2.imread('temp-images/0000.jpg')
    sharpness = tenengrad_rgb(image)
    print(f'Sharpness: {sharpness}')
