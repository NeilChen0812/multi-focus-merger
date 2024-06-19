import cv2
import numpy as np
import os
from tqdm import tqdm
from utils import dilation_erosion
from edge_detection import edge_detection_rgb
from film2image import film2image
from segmentation import contours_segmentation
from contour import find_contours
from sharpness_comparison import sharpness_comparison


def main():
    # Convert video to images
    video_path = 'videos/P1500667.MP4'
    output_frame_folder = "temp-images"
    sampling_interval = 10
    film2image(video_path, output_frame_folder, sampling_interval)

    # Read images
    image_list = [cv2.imread(os.path.join(output_frame_folder, image))
                  for image in os.listdir(output_frame_folder)]

    # Focus stacking
    sharpest_image = image_list[0]
    for i, image in enumerate(tqdm(image_list, desc='Focus stacking')):
        edges = edge_detection_rgb(image)
        contours = find_contours(dilation_erosion(edges))
        segmented_images = contours_segmentation(image, contours)
        segmented_sharpest_image = contours_segmentation(
            sharpest_image, contours)
        sharpest_parts = []
        for j, segmented_image in enumerate(segmented_images):
            sharpest_part = sharpness_comparison(
                segmented_image, segmented_sharpest_image[j])
            sharpest_parts.append(sharpest_part)

        sharpest_image = np.zeros_like(sharpest_image)
        for sharpest_part in sharpest_parts:
            sharpest_image = cv2.add(sharpest_image, sharpest_part)

    cv2.imwrite('result/sharpest_by_contour.jpg', sharpest_image)


main()
