#有換照片但沒換位置
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

def save_image(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    cv2.imwrite(os.path.join(folder, filename), image)

def main():
    folder_path = '003'  # Replace with your folder path
    
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
        
        # Get the dimensions of the corrected stitched image
        height, width = corrected_image.shape[:2]
        
        # Create a black background image with the same dimensions as corrected_image
        final_image = np.zeros_like(corrected_image)
        
        # Calculate and display the offsets for each subsequent image relative to its predecessor
        offsets = [(0, 0)]  # Offset list to store (x_offset, y_offset)
        prev_x = 0  # Initialize previous x offset
        
        # Create a folder to save images
        result_folder = 'DXT'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        
        for i in range(len(images)):
            # Compare current image with the previous image
            current_image = images[i]
            
            # Calculate offset in x direction from the previous image
            offset_x = prev_x - 0  # Start with zero offset from the first image
            
            # Update prev_x for the next image placement
            prev_x += current_image.shape[1]  # width of the current image
            
            offsets.append((offset_x, 0))  # Store the offset
            
            # Place each image in final_image according to calculated offsets
            current_x = width // 2  # Start placing from the center of the image
            
            offset_x, offset_y = offsets[i]
            
            start_y = (height - current_image.shape[0]) // 2
            end_y = start_y + current_image.shape[0]
            
            start_x = current_x + offset_x
            end_x = start_x + current_image.shape[1]
            
            # Ensure the image fits within bounds of final_image
            if start_x < 0:
                start_x = 0
                end_x = current_image.shape[1]
            elif end_x > width:
                end_x = width
                start_x = width - current_image.shape[1]
            
            final_image[start_y:end_y, start_x:end_x] = current_image
            
            # Save the current final_image with a unique filename
            save_image(final_image, result_folder, f"{i}.png")
            
            # Update current_x for the next image placement
            current_x = end_x
        
        # Resize the final image to 50% of its original size
        final_image_resized = cv2.resize(final_image, (final_image.shape[1] // 2, final_image.shape[0] // 2))
        
        # Display the resized final image with the second image in the new calculated position
        cv2.imshow('Final Image with Second Image Positioned', final_image_resized)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Close the window
        print("Processed image displayed successfully.")
    else:
        print("Failed to create panorama.")


if __name__ == "__main__":
    main()
