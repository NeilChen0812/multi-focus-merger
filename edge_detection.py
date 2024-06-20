import cv2
import numpy as np
from pykuwahara import kuwahara


def edge_detection_rgb(image, low_threshold=50, high_threshold=150, kuwa=False):
    if not isinstance(image, np.ndarray):
        raise ValueError(
            'Input must be a numpy array but got {}'.format(type(image)))
    if image.ndim != 3:
        raise ValueError('Input must be a color image')

    if kuwa:
        image = kuwahara(image, method='gaussian', radius=7)

    # b_channel, g_channel, r_channel = cv2.split(image)
    # b_edges = cv2.Canny(b_channel, low_threshold, high_threshold)
    # g_edges = cv2.Canny(g_channel, low_threshold, high_threshold)
    # r_edges = cv2.Canny(r_channel, low_threshold, high_threshold)
    # edges = cv2.bitwise_or(b_edges, g_edges)
    # edges = cv2.bitwise_or(edges, r_edges)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    return edges


if __name__ == '__main__':
    image = cv2.imread('./images/sharpest_image.jpg')
    edges = edge_detection_rgb(image)
    cv2.imwrite('./images/edges.jpg', edges)
