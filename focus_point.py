import numpy as np
from sharpness_comparison import get_sharpness_values
from film2image import film2image_list


def find_focus_points(images, sharpness_threshold=2000, diff_threshold=100):
    sharpness_values = get_sharpness_values(images)
    sharpness_values = np.array(sharpness_values)
    diff = np.insert(np.diff(sharpness_values), 0, 0)
    focus_points = np.where(
        (np.abs(diff) < diff_threshold) & (sharpness_values > sharpness_threshold))[0]
    return focus_points


if __name__ == '__main__':
    # parameters
    video_name = 'P1500718'
    video_path = f'videos/{video_name}.MP4'
    sampling_interval = 1

    images = film2image_list(video_path, sampling_interval)
    sharpness_values = get_sharpness_values(images)
    print(sharpness_values)

    focus_points = find_focus_points(sharpness_values)
    print(focus_points)
