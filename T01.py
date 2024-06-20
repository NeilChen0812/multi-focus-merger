#顯示合成圖
import cv2
import numpy as np
import os

def load_images_from_folder(folder, scale_factor=0.5):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            # Resize image to reduce memory usage
            img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))
            images.append(img)
    return images

def stitch_images(images):
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    (status, stitched) = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        return stitched
    else:
        print(f"Stitching failed with status code: {status}")
        return None

def correct_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return image

    # Find the average angle of the detected lines
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        angles.append(angle)

    average_angle = np.mean(angles)

    # Rotate the image to make lines horizontal
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, average_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def main():
    folder_path = '008'  # Replace with your folder path
    
    images = load_images_from_folder(folder_path, scale_factor=0.5)
    
    if len(images) == 0:
        print("No images found in the specified folder.")
        return
    
    if len(images) > 500:
        print("Too many images. Please provide 500 or fewer images.")
        return
    
    stitched_image = stitch_images(images)
    
    if stitched_image is not None:
        # Correct the perspective to make top and bottom edges parallel
        corrected_image = correct_perspective(stitched_image)
        
        # Resize the corrected stitched image to 50% of its original size
        corrected_image_resized = cv2.resize(corrected_image, (corrected_image.shape[1] // 2, corrected_image.shape[0] // 2))
        
        # Display the resized corrected stitched image
        cv2.imshow('Stitched Panorama', corrected_image_resized)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Close the window
        print("Panorama displayed successfully.")
    else:
        print("Failed to create panorama.")

if __name__ == "__main__":
    main()