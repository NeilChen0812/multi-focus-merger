#這是做5張
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
        (img5, img4, img3, img2, img1) = imgs
        # 获取关键点和描述符
        (kp1, des1) = self.detectAndDescribe(img1)
        (kp2, des2) = self.detectAndDescribe(img2)
        print(len(kp1), len(des1))
        print(len(kp2), len(des2))
        R = self.matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh)

        # 如果没有足够的最佳匹配点，M为None
        if R is None:
            return None
        (good, M, mask) = R
        print(M)
        # 对img1透视变换，M是ROI区域矩阵， 变换后的大小是(img1.w+img2.w, img1.h)
        result = cv.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        # 将img2的值赋给结果图像
        result[0:img2.shape[0], 0:img2.shape[1]] = img2

        # 对result进行第二次拼接
        (kp3, des3) = self.detectAndDescribe(result)
        (kp4, des4) = self.detectAndDescribe(img3)
        R2 = self.matchKeyPoints(kp3, kp4, des3, des4, ratio, reprojThresh)

        if R2 is None:
            return None
        (good2, M2, mask2) = R2
        result = cv.warpPerspective(result, M2, (result.shape[1] + img3.shape[1], result.shape[0]))
        result[0:img3.shape[0], 0:img3.shape[1]] = img3

        # 对result进行第三次拼接
        (kp5, des5) = self.detectAndDescribe(result)
        (kp6, des6) = self.detectAndDescribe(img4)
        R3 = self.matchKeyPoints(kp5, kp6, des5, des6, ratio, reprojThresh)

        if R3 is None:
            return None
        (good3, M3, mask3) = R3
        result = cv.warpPerspective(result, M3, (result.shape[1] + img4.shape[1], result.shape[0]))
        result[0:img4.shape[0], 0:img4.shape[1]] = img4

        # 对result进行第四次拼接
        (kp7, des7) = self.detectAndDescribe(result)
        (kp8, des8) = self.detectAndDescribe(img5)
        R4 = self.matchKeyPoints(kp7, kp8, des7, des8, ratio, reprojThresh)

        if R4 is None:
            return None
        (good4, M4, mask4) = R4
        result = cv.warpPerspective(result, M4, (result.shape[1] + img5.shape[1], result.shape[0]))
        result[0:img5.shape[0], 0:img5.shape[1]] = img5

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
    if len(images) < 5:
        print("Not enough images to stitch")
        return

    stitched_images = []
    for i in range(0, len(images), 5):
        imgs = images[i:i+5]
        if len(imgs) < 5:
            break
        imgs = [imutils.resize(img, width=400) for img in imgs]

        stitched = Stitcher()
        result = stitched.stitch(imgs)

        if result is None:
            print("Image stitching failed for the current set")
            continue

        stitched_images.append(result)

    if len(stitched_images) >= 5:
        print("Starting second round of stitching with stitched images.")
        show_images(stitched_images)
    else:
        if stitched_images:
            cv.imshow('Final Result', stitched_images[0])
            cv.waitKey(0)
            cv.destroyAllWindows()


def show_images(images):
    if len(images) < 5:
        print("Not enough stitched images to continue stitching")
        return

    stitched = Stitcher()
    imgs = [imutils.resize(img, width=400) for img in images[:5]]
    result = stitched.stitch(imgs)

    if result is None:
        print("Image stitching failed for the stitched images")
        return

    cv.imshow('Final Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    folder_path = '03'  # 指定包含图像的文件夹路径
    show(folder_path)
