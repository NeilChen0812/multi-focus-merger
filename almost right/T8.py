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

    def stitch(self, imgs, ratio=0.75, reprojThresh=4.0):
        print('A')

        result = imgs[0]
        for i in range(1, len(imgs)):
            next_img = imgs[i]

            # 获取关键点和描述符
            (kp1, des1) = self.detectAndDescribe(result)
            (kp2, des2) = self.detectAndDescribe(next_img)

            # 匹配关键点
            R = self.matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh)
            if R is None:
                print(f"Failed to stitch image {i} to the result")
                return None
            (good, M, mask) = R

            print(f"Transformation matrix for image {i}: {M}")

            # 对result进行透视变换
            result = cv.warpPerspective(result, M, (result.shape[1] + next_img.shape[1], result.shape[0]))
            result[0:next_img.shape[0], 0:next_img.shape[1]] = next_img
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

        print(f'Number of good matches: {len(good)}')
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
    for filename in sorted(os.listdir(folder)):  # Sort filenames to ensure order
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def show(folder):
    images = load_images_from_folder(folder)
    if len(images) < 10:
        print("Not enough images to stitch")
        return

    imgs = [imutils.resize(img, width=400) for img in images[:10]]

    stitched = Stitcher()
    result = stitched.stitch(imgs)

    if result is None:
        print("Image stitching failed")
    else:
        cv.imshow('Final Result', result)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    folder_path = '03'  # 指定包含图像的文件夹路径
    show(folder_path)
