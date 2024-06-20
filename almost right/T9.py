# -*- encoding: utf-8 -*-
"""
@Date ： 2020/10/4 18:57
@Author ： LGD
@File ：test.py
@IDE ：PyCharm
"""
import numpy as np
import cv2 as cv
import imutils
import os

class Stitcher:

    def stitch(self, imgs, ratio=0.75, reprojThresh=4.0, showMatches=False):
        print('A')
        # 将图像列表反转，确保从第一张开始拼接
        imgs.reverse()
        result = imgs[0]

        for img in imgs[1:]:
            (kp1, des1) = self.detectAndDescribe(result)
            (kp2, des2) = self.detectAndDescribe(img)
            R = self.matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh)

            if R is None:
                continue
            (good, M, mask) = R
            result = cv.warpPerspective(result, M, (result.shape[1] + img.shape[1], result.shape[0]))
            result[0:img.shape[0], 0:img.shape[1]] = img

        # 裁剪黑色边界
        result = self.crop_black_edges(result)

        return result

    def detectAndDescribe(self, img):
        print('B')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 使用SIFT检测特征
        sift = cv.SIFT_create()
        (kps, des) = sift.detectAndCompute(img, None)

        kps = np.float32([kp.pt for kp in kps])
        # 返回关键点和描述符
        return kps, des

    def matchKeyPoints(self, kp1, kp2, des1, des2, ratio, reprojThresh):
        print('C')
        # 初始化BF,因为使用的是SIFT ，所以使用默认参数
        matcher = cv.DescriptorMatcher_create('BruteForce')
        matches = matcher.knnMatch(des1, des2, 2)

        # 获取理想匹配
        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                good.append((m[0].trainIdx, m[0].queryIdx))

        print(len(good))
        # 最少要有四个点才能做透视变换
        if len(good) > 4:
            src_pts = np.float32([kp1[i] for (_, i) in good])
            dst_pts = np.float32([kp2[i] for (i, _) in good])

            # 通过两个图像的关键点计算变换矩阵
            (M, mask) = cv.findHomography(src_pts, dst_pts, cv.RANSAC, reprojThresh)

            # 返回最佳匹配点、变换矩阵和掩模
            return good, M, mask
        # 如果不满足最少四个 就返回None
        return None

    def crop_black_edges(self, img):
        # 转换为灰度图像
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 将黑色像素转换为白色（255），其他像素转换为黑色（0）
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

        # 找到所有白色（有效图像）区域的边界
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            # 获取边界框
            x, y, w, h = cv.boundingRect(contours[0])

            # 裁剪图像
            img_cropped = img[y:y+h, x:x+w]
            return img_cropped

        return img


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def show(folder):
    images = load_images_from_folder(folder)
    if len(images) == 0:
        print("No images found in the folder")
        return

    max_images_per_batch = 100
    stitched_images = []

    for i in range(0, len(images), max_images_per_batch):
        imgs = images[i:i+max_images_per_batch]
        imgs = [imutils.resize(img, width=400) for img in imgs]

        stitched = Stitcher()
        result = stitched.stitch(imgs)

        if result is None:
            print(f"Image stitching failed for images {i+1} to {min(i+max_images_per_batch, len(images))}")
            continue

        stitched_images.append(result)
        cv.imshow(f'Stitched Result {i+1} to {min(i+max_images_per_batch, len(images))}', result)

    cv.waitKey(0)
    cv.destroyAllWindows()

    if len(stitched_images) >= 2:
        print("Starting second round of stitching with stitched images.")
        show_images(stitched_images)


def show_images(images):
    if len(images) < 2:
        print("Not enough stitched images to continue stitching")
        return

    stitched = Stitcher()
    imgs = [imutils.resize(img, width=400) for img in images[:2]]
    result = stitched.stitch(imgs)

    if result is None:
        print("Image stitching failed for the stitched images")
        return

    cv.imshow('Final Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    folder_path = '06'  # 指定包含图像的文件夹路径
    show(folder_path)