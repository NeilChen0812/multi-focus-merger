import cv2
import numpy as np
from utils import dilation_erosion


def edge_detection(image):
    # 将图像拆分为 BGR 三个通道
    b_channel, g_channel, r_channel = cv2.split(image)

    # 对每个通道应用 Canny 边缘检测
    b_edges = cv2.Canny(b_channel, 50, 150)
    g_edges = cv2.Canny(g_channel, 50, 150)
    r_edges = cv2.Canny(r_channel, 50, 150)

    # 合并边缘
    edges = cv2.bitwise_or(b_edges, g_edges)
    edges = cv2.bitwise_or(edges, r_edges)

    return edges


def find_contours(edges):
    # 寻找边缘的轮廓
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个与原始图像大小相同的掩码，用于绘制轮廓
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)

    # 设置面积阈值，用于过滤小轮廓
    area_threshold = 10000  # 根据需要调整阈值

    # 遍历每个轮廓并绘制到掩码上
    filtered_contours = []
    for contour in contours:
        # 计算当前轮廓的面积
        area = cv2.contourArea(contour)

        # 如果轮廓面积大于阈值，则保留该轮廓
        if area > area_threshold:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    return filtered_contours


def contours_segmentation(image, contours):
    # 创建一个与原始图像大小相同的掩码
    height, width, _ = image.shape
    all_masks = np.zeros((height, width), dtype=np.uint8)

    # 对每个轮廓提取并显示分割区域
    for i, contour in enumerate(contours):
        # 创建一个全黑图像作为掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        # 在掩码上绘制当前轮廓
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        # 将掩码应用于原始彩色图像
        segmented_image = cv2.bitwise_and(image, image, mask=mask)

        # 显示每个分割区域
        cv2.imshow(f'Segment {i+1}', segmented_image)
        cv2.waitKey(0)

    # 处理剩余未被任何轮廓覆盖的区域
    # 将所有轮廓绘制在一起，生成完整掩码
    for contour in contours:
        cv2.drawContours(all_masks, [contour], -1, (255), thickness=cv2.FILLED)

    # 取反得到未被覆盖的区域掩码
    uncovered_mask = cv2.bitwise_not(all_masks)

    # 将掩码应用于原始彩色图像
    uncovered_regions = cv2.bitwise_and(image, image, mask=uncovered_mask)

    # 显示剩余未被覆盖的区域
    cv2.imshow('Uncovered Regions', uncovered_regions)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread('./result/sharpest_01.jpg')
    edges = edge_detection(image)
    cv2.imwrite('./images/edges.jpg', edges)
    contours = find_contours(dilation_erosion(edges))
    # contours = find_contours(edges)
    print(f'Number of Contours: {len(contours)}')
    cv2.imwrite('./images/contours.jpg', image)
    segmented_image = contours_segmentation(image, contours)
