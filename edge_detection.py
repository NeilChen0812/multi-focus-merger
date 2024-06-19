import cv2
import numpy as np
from utils import dilation_erosion


def edge_detection_rgb(image):
    if not isinstance(image, np.ndarray):
        raise ValueError('Input must be a numpy array')
    if image.ndim != 3:
        raise ValueError('Input must be a color image')
    if image.shape[2] != 3:
        raise ValueError('Input must have 3 color channels')

    b_channel, g_channel, r_channel = cv2.split(image)

    b_edges = cv2.Canny(b_channel, 50, 150)
    g_edges = cv2.Canny(g_channel, 50, 150)
    r_edges = cv2.Canny(r_channel, 50, 150)

    edges = cv2.bitwise_or(b_edges, g_edges)
    edges = cv2.bitwise_or(edges, r_edges)

    return edges


def find_contours(edges, area_threshold=10000):
    if not isinstance(area_threshold, int):
        raise ValueError('Area threshold must be an integer')
    if not isinstance(edges, np.ndarray):
        raise ValueError('Input must be a numpy array')
    if edges.ndim != 2:
        raise ValueError('Input must be a grayscale image')

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            filtered_contours.append(contour)

    return filtered_contours


def contours_segmentation(image, contours):
    if not isinstance(image, np.ndarray) or not isinstance(contours, list):
        raise ValueError('Input must be a numpy array and a list of contours')
    if contours and not all(isinstance(contour, np.ndarray) for contour in contours):
        raise ValueError('All contours must be numpy arrays')

    segmented_images = []
    height, width, _ = image.shape
    all_masks = np.zeros((height, width), dtype=np.uint8)
    contours_count = len(contours) + 1

    for i, contour in enumerate(contours):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(all_masks, [contour], -1, (255), thickness=cv2.FILLED)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        segmented_image = cv2.bitwise_and(image, image, mask=mask)

        segmented_images.append(segmented_image)
        print("\rContours segmentation: {}/{}".format(i+1, contours_count), end='')

    uncovered_mask = cv2.bitwise_not(all_masks)
    uncovered_regions = cv2.bitwise_and(image, image, mask=uncovered_mask)
    segmented_images.append(uncovered_regions)
    print("\rContours segmentation: {}/{}".format(contours_count, contours_count), end='')
    print("  (Done)")

    return segmented_images


if __name__ == '__main__':
    image = cv2.imread('./result/sharpest_01.jpg')
    edges = edge_detection_rgb(image)
    print(f'Edges Shape: {edges.shape}')
    cv2.imwrite('./images/edges.jpg', edges)
    contours = find_contours(dilation_erosion(edges))
    # contours = find_contours(edges)
    print(f'Number of Contours: {len(contours)}')
    cv2.imwrite('./images/contours.jpg', image)
    segmented_image = contours_segmentation(image, contours)
