import cv2
import numpy as np


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


if __name__ == '__main__':
    from utils import dilation_erosion
    from edge_detection import edge_detection_rgb

    image = cv2.imread('./result/sharpest_01.jpg')
    edges = edge_detection_rgb(image)
    print(f'Edges Shape: {edges.shape}')
    cv2.imwrite('./images/edges.jpg', edges)
    contours = find_contours(dilation_erosion(edges))
    # contours = find_contours(edges)
    print(f'Number of Contours: {len(contours)}')
    cv2.imwrite('./images/contours.jpg', image)
