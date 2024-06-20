import cv2
import numpy as np


def find_contours(edges, area_threshold=5000):
    if not isinstance(area_threshold, int):
        raise ValueError('Area threshold must be an integer but got {}'.format(
            type(area_threshold)))
    if not isinstance(edges, np.ndarray):
        raise ValueError('Input must be a numpy array but got {}'.format(
            type(edges)))
    if edges.ndim != 2:
        raise ValueError('Input must be a binary image')

    # find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours by area
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            filtered_contours.append(contour)

    return filtered_contours


if __name__ == '__main__':
    from edge_detection import edge_detection_rgb

    image = cv2.imread('./result/sharpest_01.jpg')
    edges = edge_detection_rgb(image)
    print(f'Edges Shape: {edges.shape}')
    cv2.imwrite('./images/edges.jpg', edges)
    kernel_size = 7
    contours = find_contours(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones(
        (kernel_size, kernel_size), np.uint8)), area_threshold=1000)
    # contours = find_contours(edges)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    print(f'Number of Contours: {len(contours)}')
    cv2.imwrite('./images/contours.jpg', image)
