import cv2
import os
from tenengrad import tenengrad_rgb


def image_sharpness_comparison(image_folder):
    image_folder = 'temp-images'
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


if __name__ == '__main__':
    image_folder = 'temp-images'
    sharpest_image = image_sharpness_comparison(image_folder)
    cv2.imwrite('./images/sharpest_image.jpg', sharpest_image)
