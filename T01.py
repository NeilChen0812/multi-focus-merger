#顯示合成圖(最快版本)
import cv2
import numpy as np
import os

# 函數：從資料夾中讀取圖像並進行縮放
def load_images_from_folder(folder, scale_factor=0.5):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))  # 讀取圖像
        if img is not None:
            # 縮放圖像以減少記憶體使用量
            img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))
            images.append(img)  # 將處理後的圖像加入列表中
    return images

# 函數：將多張圖像拼接成全景圖像
def stitch_images(images):
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)  # 創建拼接器實例
    (status, stitched) = stitcher.stitch(images)  # 執行圖像拼接
    
    if status == cv2.Stitcher_OK:
        return stitched  # 如果成功，返回拼接後的圖像
    else:
        print(f"Stitching failed with status code: {status}")  # 拼接失敗時顯示錯誤信息
        return None

# 函數：校正透視，使圖像的頂部和底部邊緣平行
def correct_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 將圖像轉為灰度
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # 檢測圖像邊緣
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)  # 使用霍夫變換檢測直線

    if lines is None:
        return image  # 如果未檢測到直線，返回原始圖像

    # 計算檢測到直線的平均角度
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90  # 計算角度並調整為相對於垂直的角度
        angles.append(angle)

    average_angle = np.mean(angles)  # 計算平均角度

    # 將圖像旋轉使得直線變為水平
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, average_angle, 1.0)  # 獲取旋轉矩陣
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)  # 進行圖像旋轉

    return rotated  # 返回校正後的圖像

# 主函數：程式的入口
def main():
    folder_path = '003'  # 設置資料夾路徑，用來存放待拼接的圖像
    
    images = load_images_from_folder(folder_path, scale_factor=0.5)  # 從資料夾中讀取圖像並進行縮放
    
    if len(images) == 0:
        print("指定資料夾中未找到圖像。")
        return
    
    if len(images) > 500:
        print("圖像過多。請提供少於500張圖像。")
        return
    
    stitched_image = stitch_images(images)  # 將圖像拼接成全景圖像
    
    if stitched_image is not None:
        corrected_image = correct_perspective(stitched_image)  # 校正全景圖像的透視效果
        
        # 縮小校正後的全景圖像為原始尺寸的50%
        corrected_image_resized = cv2.resize(corrected_image, (corrected_image.shape[1] // 2, corrected_image.shape[0] // 2))
        
        # 顯示縮小後的校正全景圖像
        cv2.imshow('Stitched Panorama', corrected_image_resized)
        cv2.waitKey(0)  # 等待按鍵按下來關閉窗口
        cv2.destroyAllWindows()  # 關閉窗口
        print("全景圖像顯示成功。")
    else:
        print("無法創建全景圖像。")

# 程式的入口點，執行主函數
if __name__ == "__main__":
    main()
