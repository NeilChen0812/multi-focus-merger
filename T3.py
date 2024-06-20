# 稍微快一點的版本
# 實測50張5分鐘 60張7分鐘
# 找後面有些片段是否重複可以不用
import cv2
import numpy as np
import os

# 設定圖片所在的資料夾路徑
folder_path = '06'

# 列出資料夾中所有的圖片文件
image_files = [os.path.join(folder_path, f) for f in os.listdir(
    folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# 確定文件按照名稱排序
image_files.sort()


# 讀取圖片
images = []
for filename in image_files[:60]:  # 只讀取前50張圖片
    img = cv2.imread(filename)
    images.append(img)

# 創建拼接器
stitcher = cv2.Stitcher_create()

# 拼接圖片
status, result_image = stitcher.stitch(images)


# 檢查拼接是否成功
if status == cv2.Stitcher_OK:
    # 顯示拼接後的全景圖片
    cv2.namedWindow('Panorama', cv2.WINDOW_NORMAL)

    # 計算50%大小
    scale_percent = 50  # 百分比大小
    width = int(result_image.shape[1] * scale_percent / 100)
    height = int(result_image.shape[0] * scale_percent / 100)

    # 調整視窗大小以顯示50%大小的圖片
    cv2.resizeWindow('Panorama', width, height)

    cv2.imshow('Panorama', result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error during stitching:", status)
