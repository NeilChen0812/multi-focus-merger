import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tenengrad import tenengrad_rgb
from film2image import film2image_list
from sharpness_comparison import get_sharpness_values


def test():
    # parameters
    video_name = 'P1500718'
    video_path = f'videos/{video_name}.MP4'
    sampling_interval = 1
    reverse = False  # whether to process the video in reverse order
    # load video
    vc = cv2.VideoCapture(video_path)
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    sharpness_values = []

    # process video
    if reverse:
        start = frame_count
        end = 0
        sampling_interval *= -1
    else:
        start = 0
        end = frame_count

    for idx in tqdm(range(start, end, sampling_interval), desc='Processing frames'):
        vc.set(1, idx)
        ret, frame = vc.read()
        if frame is not None:
            sharpness = tenengrad_rgb(frame)
            sharpness_values.append(sharpness)

    plt.figure()
    plt.plot(sharpness_values, marker='o', linestyle='-', color='b')
    plt.title('Sharpness')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()


def draw_curve(data):
    plt.figure()
    plt.plot(data, marker='o', linestyle='-', color='b')
    plt.title('Sharpness')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # parameters
    video_name = 'P1500667'
    video_path = f'videos/{video_name}.MP4'
    sampling_interval = 1

    images = film2image_list(video_path, sampling_interval)
    sharpness_values = get_sharpness_values(images)

    draw_curve(sharpness_values)
    draw_curve(np.diff(sharpness_values))
