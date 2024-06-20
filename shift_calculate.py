import cv2
import numpy as np


def calculate_translation(img1, img2):

    # 初始化ORB检测器
    orb = cv2.ORB_create()

    # 检测并计算特征点和描述符
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # 创建BFMatcher对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 匹配描述符
    matches = bf.match(descriptors1, descriptors2)

    # 按照距离排序匹配结果
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配的关键点
    src_pts = np.float32(
        [keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32(
        [keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # 计算平移向量
    translations = dst_pts - src_pts

    # 计算平均平移
    mean_translation = np.mean(translations, axis=0)

    return mean_translation


# 调用函数计算平移向量
folder_path = "008/"

translation_vector = calculate_translation(img1, img2)
print(f"Translation vector: {translation_vector}")
