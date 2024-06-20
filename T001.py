import cv2
import numpy as np
import os


def load_images_from_folder(folder, scale_factor=0.5):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            # Resize image to reduce memory usage
            img = cv2.resize(
                img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))
            images.append((filename, img))
    return images


def stitch_images(images):
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    (status, stitched) = stitcher.stitch([img for filename, img in images])

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
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def main():
    folder_path = '003'  # Replace with your folder path

    images = load_images_from_folder(folder_path, scale_factor=0.5)

    if len(images) == 0:
        print("No images found in the specified folder.")
        return

    if len(images) > 500:
        print("Too many images. Please provide 500 or fewer images.")
        return

    # Create AAA folder if it doesn't exist
    output_folder = 'AAA'
    os.makedirs(output_folder, exist_ok=True)

    stitched_image = stitch_images(images)

    if stitched_image is not None:
        # Correct the perspective to make top and bottom edges parallel
        corrected_image = correct_perspective(stitched_image)

        # Get the dimensions of the corrected stitched image
        height, width = corrected_image.shape[:2]

        # Create a black background image with the same dimensions as corrected_image
        final_image = np.zeros_like(corrected_image, dtype=np.uint8)

        # Calculate where to place the first image in the center
        start_x = (width - images[0][1].shape[1]) // 2
        start_y = (height - images[0][1].shape[0]) // 2

        # Place the first image in the calculated position on the black background
        final_image[start_y:start_y + images[0][1].shape[0],
                    start_x:start_x + images[0][1].shape[1]] = images[0][1]

        # Display and save the first image
        print(
            f"File: {images[0][0]}, Image size: {images[0][1].shape}, Background size: {final_image.shape}")
        cv2.imshow('Aligned Image 1', final_image)
        cv2.imwrite(os.path.join(output_folder, '1.jpg'), final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Processed image 1 displayed and saved successfully.")

        # For the rest of the images
        for i in range(1, len(images)):
            # Calculate the offset in x direction from the previous image
            offset_x = start_x - (width - images[i][1].shape[1]) // 2

            # Check if the placement is valid
            if offset_x < 0 or offset_x + images[i][1].shape[1] > width or start_y + images[i][1].shape[0] > height:
                print(
                    f"Image {images[i][0]} cannot be placed due to size mismatch.")
                print(
                    f"Image size: {images[i][1].shape}, Offset x: {offset_x}, Background width: {width}, Background height: {height}")
                continue

            # Create a new black background image
            final_image = np.zeros_like(corrected_image, dtype=np.uint8)

            # Place the current image in the calculated position on the black background
            final_image[start_y:start_y + images[i][1].shape[0],
                        start_x:start_x + images[i][1].shape[1]] = images[i][1]

            # Fill the regions that do not overlap with black
            final_image[:start_y, :] = 0
            final_image[start_y + images[i][1].shape[0]:, :] = 0
            final_image[:, :start_x] = 0
            final_image[:, start_x + images[i][1].shape[1]:] = 0

            # Display and save the current image
            print(
                f"File: {images[i][0]}, Image size: {images[i][1].shape}, Background size: {final_image.shape}")
            cv2.imshow(f'Aligned Image {i + 1}', final_image)
            cv2.imwrite(os.path.join(output_folder,
                        f'{i + 1}.jpg'), final_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(f"Processed image {i + 1} displayed and saved successfully.")

            # Print current background size
            print(
                f"Current background size after image {i + 1}: {final_image.shape}")

            # Update start_x for the next image placement
            start_x += images[i][1].shape[1]

    else:
        print("Failed to create panorama.")


if __name__ == "__main__":
    main()
