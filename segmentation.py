import cv2
import numpy as np


def contours_segmentation(image, contours, area_threshold=1000):
    if not isinstance(image, np.ndarray):
        raise ValueError(
            'Input must be a numpy array but got {}'.format(type(image)))
    if not isinstance(contours, list):
        raise ValueError(
            'Input must be a list of contours but got {}'.format(type(contours)))
    if contours and not all(isinstance(contour, np.ndarray) for contour in contours):
        raise ValueError('All contours must be numpy arrays')

    # set up
    segmented_images = []
    height, width, _ = image.shape
    all_masks = np.zeros((height, width), dtype=np.uint8)

    # segment the image based on the contours
    for i, contour in enumerate(contours):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(all_masks, [contour], -1, (255), thickness=cv2.FILLED)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        segmented_image = cv2.bitwise_and(image, image, mask=mask)
        segmented_images.append(segmented_image)

    # add the uncovered regions
    uncovered_mask = cv2.bitwise_not(all_masks)
    num_labels, labels = cv2.connectedComponents(uncovered_mask)
    for label in range(1, num_labels):
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[labels == label] = 255
        if np.sum(mask) >= 255 * area_threshold:
            uncovered_regions = cv2.bitwise_and(image, image, mask=mask)
            segmented_images.append(uncovered_regions)

    return segmented_images


if __name__ == '__main__':
    from edge_detection import edge_detection_rgb
    from contour import find_contours

    image = cv2.imread('./result/sharpest_01.jpg')
    edges = edge_detection_rgb(image)
    print(f'Edges Shape: {edges.shape}')
    cv2.imwrite('./images/edges.jpg', edges)
    kernel_size = 7
    contours = find_contours(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones(
        (kernel_size, kernel_size), np.uint8)))
    # contours = find_contours(edges)
    print(f'Number of Contours: {len(contours)}')
    cv2.imwrite('./images/contours.jpg', image)
    segmented_image = contours_segmentation(image, contours)
