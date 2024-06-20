import cv2
import numpy as np
import os
from tqdm import tqdm
from film2image import film2image


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
filename = "P1500674"
folder_path = f"_output-images_/{filename}"
video_path = f'videos/{filename}.MP4'

film2image(video_path, f"_output-images_/{filename}", 1)

images = [cv2.imread(f"{folder_path}/{img}")
          for img in os.listdir(folder_path)]

translation_vectors = []
for i in tqdm(range(len(images) - 2), desc="Calculating translation vectors"):
    translation_vector = calculate_translation(images[i], images[i + 1])
    translation_vectors.append(translation_vector)

# 计算累积平移向量
cumulative_translation = []
default_translation = np.array([0, 0])
for i, vector in enumerate(translation_vectors):
    default_translation = default_translation + vector
    cumulative_translation.append(
        [int(default_translation[0]), int(default_translation[1])])

max_x = max([abs(vector[0]) for vector in cumulative_translation])
max_y = max([abs(vector[1]) for vector in cumulative_translation])
min_x = min([vector[0] for vector in cumulative_translation])
min_y = min([vector[1] for vector in cumulative_translation])
height = max_y - min_y + images[0].shape[0]
width = max_x - min_x + images[0].shape[1]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
file_name = f"translation-{folder_path}.mp4"
if os.path.exists(file_name):
    os.remove(file_name)

video = cv2.VideoWriter(
    file_name, fourcc, 50, (width, height))

for i in tqdm(range(len(cumulative_translation)), desc="Creating video"):
    image = images[i+1]
    x_translation = cumulative_translation[i][0]
    y_translation = cumulative_translation[i][1]
    translated_image = np.zeros((height, width, 3), dtype=np.uint8)
    translated_image[max_y-y_translation:max_y-y_translation+image.shape[0],
                     max_x-x_translation:max_x-x_translation+image.shape[1]] = image
    video.write(translated_image)

video.release()
