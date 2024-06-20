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
from concurrent import futures

class Stitcher:

    def stitch(self, imgs, ratio=0.75, reprojThresh=4.0, showMatches=False):
        print('A')
        if len(imgs) < 2:
            print("At least two images are required for stitching")
            return None

        # 将图像列表转换为灰度图像和特征描述符
        gray_imgs = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in imgs]
        kps, descs = zip(*[self.detectAndDescribe(img) for img in gray_imgs])

        R = self.matchKeyPoints(kps[0], descs[0], kps[1], descs[1], ratio, reprojThresh)

        if R is None:
            return None

        (good, M, mask) = R
        result = cv.warpPerspective(imgs[0], M, (imgs[0].shape[1] + imgs[1].shape[1], imgs[0].shape[0]))
        result[0:imgs[1].shape[0], 0:imgs[1].shape[1]] = imgs[1]

        # 逐一拼接剩余图像
        for i in range(2, len(imgs)):
            kps_i, descs_i = self.detectAndDescribe(gray_imgs[i])
            R = self.matchKeyPoints(kps_i, descs_i, kps[i-1], descs[i-1], ratio, reprojThresh)
            if R is None:
                continue
            (good, M, mask) = R
            result = cv.warpPerspective(result, M, (result.shape[1] + imgs[i].shape[1], result.shape[0]))
            result[0:imgs[i].shape[0], 0:imgs[i].shape[1]] = imgs[i]

        result = self.crop_black_edges(result)

        if showMatches:
            vis = self.drawMatches(imgs[0], imgs[1], kps[0], good, mask)
            return result, vis

        return result

    def detectAndDescribe(self, img):
        print('B')
        sift = cv.SIFT_create()
        (kps, des) = sift.detectAndCompute(img, None)
        kps = np.float32([kp.pt for kp in kps])
        return kps, des

    def matchKeyPoints(self, kp1, des1, kp2, des2, ratio, reprojThresh):
        print('C')
        matcher = cv.DescriptorMatcher_create('BruteForce')
        matches = matcher.knnMatch(des1, des2, 2)

        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, reprojThresh)
            return good, M, mask

        return None

    def drawMatches(self, img1, img2, kp1, matches, mask):
        print('D')
        (hA, wA) = img1.shape[:2]
        (hB, wB) = img2.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')
        vis[0:hA, 0:wA] = img1
        vis[0:hB, wA:] = img2
        for m in matches:
            if mask[m]:
                ptA = (int(kp1[m.queryIdx][0]), int(kp1[m.queryIdx][1]))
                ptB = (int(kp1[m.trainIdx][0]) + wA, int(kp1[m.trainIdx][1]))
                cv.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis

    def crop_black_edges(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv.boundingRect(contours[0])
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
    if len(images) < 2:
        print("At least two images are required for stitching")
        return

    stitched_images = []
    stitched = Stitcher()

    with futures.ThreadPoolExecutor() as executor:
        results = executor.map(stitched.stitch, [images[i:i+2] for i in range(len(images)-1)])

    for result in results:
        if result is not None:
            stitched_images.append(result)

    if len(stitched_images) > 0:
        cv.imshow('Final Result', stitched_images[-1])
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    folder_path = '003'  # 指定包含图像的文件夹路径
    show(folder_path)
