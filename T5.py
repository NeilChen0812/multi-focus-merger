# 網路找的，效果好但速度慢，還只能和兩張
import cv2
import numpy as np
import os

# Function to detect keypoints and descriptors using SIFT


def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray_image, None)
    return kp, des

# Function to get good matches between two sets of descriptors


def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

# Function to draw matches between keypoints of two images


def drawMatches(image1, kp1, image2, kp2, matches):
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    vis[0:h1, 0:w1] = image1
    vis[0:h2, w1:] = image2
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        cv2.circle(vis, (int(x1), int(y1)), 4, (0, 255, 0), 1)
        cv2.circle(vis, (int(x2) + w1, int(y2)), 4, (0, 255, 0), 1)
        cv2.line(vis, (int(x1), int(y1)),
                 (int(x2) + w1, int(y2)), (0, 255, 0), 1)
    return vis

# Function to create panoramic image


def create_panorama(images):
    # Ensure images are grayscale and get keypoints and descriptors
    keypoints = []
    descriptors = []
    for image in images:
        kp, des = sift_kp(image)
        keypoints.append(kp)
        descriptors.append(des)

    # Get good matches between consecutive image pairs
    good_matches = []
    for i in range(len(images) - 1):
        matches = get_good_match(descriptors[i], descriptors[i+1])
        good_matches.append(matches)

    # Initialize variables for perspective transformation
    ransacReprojThreshold = 4
    cumulative_H = np.eye(3, 3, dtype=np.float32)
    warped_images = [images[0]]

    # Warp images and accumulate transformation matrix
    for i in range(len(images) - 1):
        kp1 = keypoints[i]
        kp2 = keypoints[i+1]
        img1 = warped_images[-1]
        img2 = images[i+1]

        ptsA = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches[i]]).reshape(-1, 1, 2)
        ptsB = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches[i]]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(
            ptsB, ptsA, cv2.RANSAC, ransacReprojThreshold)
        cumulative_H = np.dot(cumulative_H, H)

        warped_img = cv2.warpPerspective(
            img2, cumulative_H, (img1.shape[1] + img2.shape[1], img2.shape[0]))
        warped_img[0:img1.shape[0], 0:img1.shape[1]] = img1
        warped_images.append(warped_img)

    return warped_images[-1]


# Read images from folder
folder_path = r'04'
images = []
for filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)
    if img is not None:
        images.append(img)

# Create panoramic image
panorama = create_panorama(images)

# Display the panoramic image
cv2.imshow('Panorama', panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
