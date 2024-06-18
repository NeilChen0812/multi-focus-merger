import cv2
import os
import math
from tenengrad import tenengrad_rgb


def sharpness_comparison(image_folder):
    sharpest_image = None
    sharpest_value = 0

    for img in os.listdir(image_folder):
        image_path = os.path.join(image_folder, img)
        image = cv2.imread(image_path)
        sharpness_value = tenengrad_rgb(image)
        if sharpness_value > sharpest_value:
            print(f'Sharpest Image: {image_path}')
            print(f'Sharpest Value: {sharpest_value}')
            sharpest_value = sharpness_value
            sharpest_image = image
    return sharpest_image


def partial_sharpness_comparison(image_folder, part_size=100):
    sharpest_image = cv2.imread(
        f'{image_folder}/{os.listdir(image_folder)[0]}')
    height, width, _ = sharpest_image.shape
    num_vertical_segments = math.ceil(height/part_size)
    num_horizontal_segments = math.ceil(width/part_size)
    sharpest_value = [[0 for i in range(num_horizontal_segments)]
                      for j in range(num_vertical_segments)]

    for img in os.listdir(image_folder):
        image_path = os.path.join(image_folder, img)
        image = cv2.imread(image_path)
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
    image_folder = 'temp-images'
    sharpest_image = partial_sharpness_comparison(image_folder, 50)
    cv2.imwrite('./images/partial_sharpest_image.jpg', sharpest_image)
