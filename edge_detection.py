import cv2


def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


if __name__ == '__main__':
    image = cv2.imread('./images/sharpest_image.jpg')
    edges = edge_detection(image)
    cv2.imwrite('./images/edges.jpg', edges)
