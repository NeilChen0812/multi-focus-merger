import cv2
import numpy as np


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


if __name__ == '__main__':
    image = cv2.imread('./result/sharpest_01.jpg')
    edges = edge_detection_rgb(image)
    print(f'Edges Shape: {edges.shape}')
    cv2.imwrite('./images/edges.jpg', edges)
