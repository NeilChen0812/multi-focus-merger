import os
import shutil


def copy_every_second_image(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    images = [img for img in os.listdir(source_dir) if img.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    images.sort()  # Optional: Sort the images if needed

    for idx, image in enumerate(images):
        if idx % 5 == 0:  # Every second image
            src_path = os.path.join(source_dir, image)
            dest_path = os.path.join(target_dir, image)
            shutil.copy2(src_path, dest_path)
            print(f"Copied {image} to {target_dir}")


source_directory = '002'  # Replace with the path to your source directory
target_directory = '008'  # Replace with the path to your target directory

copy_every_second_image(source_directory, target_directory)
