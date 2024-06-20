import cv2
import os
import math
import numpy as np
from tenengrad import tenengrad_rgb


def sharpness_comparison(image1, image2):
    if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
        raise ValueError('Input images must be numpy arrays but got {} and {}'.format(
            type(image1), type(image2)))

    image1_sharpness = tenengrad_rgb(image1)
    image2_sharpness = tenengrad_rgb(image2)
    if image1_sharpness > image2_sharpness:
        return image1
    else:
        return image2


def sharpness_comparison_multi_image(image_list):
    if not isinstance(image_list, list):
        raise ValueError(
            'Input must be a list of images but got {}'.format(type(image_list)))
    if not all(isinstance(image, np.ndarray) for image in image_list):
        raise ValueError(
            'All images in list must be numpy arrays but got {}'.format(
                type(image_list[0])))

    sharpest_image = None
    sharpest_value = 0

    for i, image in enumerate(image_list):
        sharpness_value = tenengrad_rgb(image)
        if sharpness_value > sharpest_value:
            sharpest_value = sharpness_value
            sharpest_image = image
    return sharpest_image


def partial_sharpness_comparison(image_list, part_size=100):
    if not isinstance(part_size, int):
        raise ValueError(
            'Part size must be an integer but got {}'.format(type(part_size)))
    if not isinstance(image_list, list):
        raise ValueError(
            'Input must be a list of images but got {}'.format(type(image_list)))
    if not all(isinstance(image, np.ndarray) for image in image_list):
        raise ValueError(
            'All images in list must be numpy arrays')

    sharpest_image = image_list[0]
    height, width, _ = sharpest_image.shape
    num_vertical_segments = math.ceil(height/part_size)
    num_horizontal_segments = math.ceil(width/part_size)
    sharpest_value = [[0 for i in range(num_horizontal_segments)]
                      for j in range(num_vertical_segments)]

    for image in enumerate(image_list):
        for i in range(num_vertical_segments):
            for j in range(num_horizontal_segments):
                x_start, x_end = j*part_size, (j+1)*part_size
                y_start, y_end = i*part_size, (i+1)*part_size
                if i == num_vertical_segments-1:
                    y_end = height
                if j == num_horizontal_segments-1:
                    x_end = width
                part = image[y_start:y_end, x_start:x_end]
                sharpness_value = tenengrad_rgb(part)
                if sharpness_value > sharpest_value[i][j]:
                    sharpest_value[i][j] = sharpness_value
                    sharpest_image[y_start:y_end, x_start:x_end] = part
    return sharpest_image


if __name__ == '__main__':
    image_folder = '_temp-images-frame_'
    image_list = [cv2.imread(os.path.join(image_folder, image))
                  for image in os.listdir(image_folder)]
    sharpest_image = sharpness_comparison_multi_image(image_list)
    cv2.imwrite('./images/sharpest_image.jpg', sharpest_image)
