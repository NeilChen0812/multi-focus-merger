import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
from edge_detection import edge_detection_rgb
from segmentation import contours_segmentation
from contour import find_contours
from sharpness_comparison import sharpness_comparison
from focus_point import find_focus_points
from film2image import film2image_list


def main():
    # parameters
    video_name = 'P1500718'
    video_path = f'videos/{video_name}.MP4'
    video_name = video_name + '_v2'
    sampling_interval = 1
    reverse = False  # whether to process the video in reverse order
    sharpness_threshold = 2000  # sharpness threshold for focus points
    diff_threshold = 100  # difference threshold for focus points
    area_threshold = 5000  # contour will be ignored if area < area_threshold
    kernel_size = 5  # cv2.morphologyEx parameter
    low_threshold, high_threshold = 50, 150  # cv2.Canny parameter
    kuwa = False  # whether to apply Kuwahara filter before edge detection

    frame_rate = 15

    # create output folder
    output_folder = f'_output-images_/{video_name}'
    if os.path.exists(output_folder):
        print("Delete old result in folder: {}".format(output_folder))
        shutil.rmtree(output_folder)
    else:
        print("Create new folder: {}".format(output_folder))
    os.makedirs(output_folder)

    # load video
    images = film2image_list(video_path, sampling_interval)

    # find focus points
    focus_points = find_focus_points(
        images, sharpness_threshold=sharpness_threshold, diff_threshold=diff_threshold)

    if reverse:
        images = images[::-1]
        focus_points = focus_points[::-1]

    sharpest_image = images[focus_points[0]].copy()

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_edge = cv2.VideoWriter(
        f'{output_folder}/edge.mp4', fourcc, frame_rate, (sharpest_image.shape[1], sharpest_image.shape[0]))
    video_dila = cv2.VideoWriter(
        f'{output_folder}/dila.mp4', fourcc, frame_rate, (sharpest_image.shape[1], sharpest_image.shape[0]))
    video_contour = cv2.VideoWriter(
        f'{output_folder}/contour.mp4', fourcc, frame_rate, (sharpest_image.shape[1], sharpest_image.shape[0]))
    video_sharpest = cv2.VideoWriter(
        f'{output_folder}/sharpest.mp4', fourcc, frame_rate, (sharpest_image.shape[1], sharpest_image.shape[0]))

    # process frames
    for idx in tqdm(focus_points, desc='Processing frames'):
        frame = images[idx].copy()
        # edge detection
        edge = edge_detection_rgb(
            frame, low_threshold=low_threshold, high_threshold=high_threshold, kuwa=kuwa)

        # dilation and erosion
        main_edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones(
            (kernel_size, kernel_size), np.uint8))

        # find contours
        contours = find_contours(main_edge, area_threshold)

        # segment the image based on the contours
        segmented_images = contours_segmentation(frame, contours)
        segmented_sharpest_image = contours_segmentation(
            sharpest_image, contours)

        # compare the sharpness of the segmented images
        sharpest_parts = []
        for j, segmented_image in enumerate(segmented_images):
            sharpest_part = sharpness_comparison(
                segmented_image, segmented_sharpest_image[j])
            sharpest_parts.append(sharpest_part)

        # combine the sharpest parts
        sharpest_image = np.zeros_like(sharpest_image)
        for sharpest_part in sharpest_parts:
            sharpest_image = cv2.add(sharpest_image, sharpest_part)

        # save images processing result
        file_name = '{}/frame{:04d}.jpg'.format(output_folder, idx)
        cv2.imwrite(file_name, frame)
        file_name = '{}/edge{:04d}.jpg'.format(output_folder, idx)
        cv2.imwrite(file_name, edge)
        file_name = '{}/dila{:04d}.jpg'.format(output_folder, idx)
        cv2.imwrite(file_name, main_edge)
        file_name = '{}/contour{:04d}.jpg'.format(output_folder, idx)
        cv2.imwrite(file_name, cv2.drawContours(
            frame.copy(), contours, -1, (0, 255, 0), 3))
        file_name = '{}/sharpest{:04d}.jpg'.format(output_folder, idx)
        cv2.imwrite(file_name, sharpest_image)

        # save result video
        video_edge.write(cv2.merge((edge, edge, edge)))
        video_dila.write(cv2.merge((main_edge, main_edge, main_edge)))
        video_contour.write(cv2.drawContours(
            frame.copy(), contours, -1, (0, 255, 0), 3))
        video_sharpest.write(sharpest_image)

    # save the sharpest image
    video_edge.release()
    video_dila.release()
    video_contour.release()
    video_sharpest.release()
    cv2.imwrite(f'result/{video_name}.jpg', sharpest_image)


main()
