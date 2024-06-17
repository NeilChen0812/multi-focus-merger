#更多照片去拚出完整的(手動挑)
import cv2
import os
import threading

def stitch_images(images):
    # 初始化Stitcher
    stitcher = cv2.Stitcher_create()
    # 尝试拼接照片
    status, stitched = stitcher.stitch(images)
    
    # 检查是否成功拼接
    if status == cv2.Stitcher_OK:
        return stitched
    else:
        print("Error during stitching")
        return None

def process_image(filepath, resized_images, lock):
    image = cv2.imread(filepath)
    if image is not None:
        # 調整照片大小
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
        # 加入共享列表
        with lock:
            resized_images.append(image)
    else:
        print(f"Failed to load image: {filename}")

# 資料夾路徑和最大照片數量
folder_path = '06'
max_images = 100  # 設定照片數量上限

# 讀取資料夾中的照片
resized_images = []
count = 0
lock = threading.Lock()

threads = []

for filename in os.listdir(folder_path):
    if count >= max_images:
        break
    
    filepath = os.path.join(folder_path, filename)
    if os.path.isfile(filepath):
        # 創建並啟動線程處理圖像
        thread = threading.Thread(target=process_image, args=(filepath, resized_images, lock))
        thread.start()
        threads.append(thread)
        count += 1

# 等待所有線程完成
for thread in threads:
    thread.join()

# 拼接照片
result_image = stitch_images(resized_images)

# 顯示拼接後的全景圖
if result_image is not None:
    cv2.imshow('Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to stitch images.")
