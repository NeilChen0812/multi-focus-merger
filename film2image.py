import os
import shutil
import cv2
from tqdm import tqdm


def film2image(video_path, output_folder="output", sampling_interval=1):
    if not isinstance(sampling_interval, int):
        raise ValueError("Sampling interval must be an integer")
    if not isinstance(output_folder, str):
        raise ValueError("Output folder must be a string")
    if not os.path.exists(video_path):
        raise FileNotFoundError("Video file not found: {}".format(video_path))

    # clear and create output folder
    if os.path.exists(output_folder):
        print("Delete old result in folder: {}".format(output_folder))
        shutil.rmtree(output_folder)
    else:
        print("Create new folder: {}".format(output_folder))
    os.makedirs(output_folder)

    # load video
    vc = cv2.VideoCapture(video_path)
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    # process video
    for idx in tqdm(range(0, frame_count, sampling_interval), desc='Film to images'):
        vc.set(1, idx)
        ret, frame = vc.read()
        if frame is not None:
            file_name = '{}/{:04d}.jpg'.format(output_folder, idx)
            cv2.imwrite(file_name, frame)

    vc.release()


def film2image_list(video_path, sampling_interval=1):
    if not isinstance(sampling_interval, int):
        raise ValueError("Sampling interval must be an integer")
    if not os.path.exists(video_path):
        raise FileNotFoundError("Video file not found: {}".format(video_path))

    # load video
    vc = cv2.VideoCapture(video_path)
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    image_list = []

    # process video
    for idx in tqdm(range(0, frame_count, sampling_interval), desc='Film to images'):
        vc.set(1, idx)
        ret, frame = vc.read()
        if frame is not None:
            image_list.append(frame)
    vc.release()

    return image_list


if __name__ == '__main__':
    filename = "P1500718"
    video_path = f'videos/{filename}.MP4'
    film2image(video_path, f"_temp-images-frame_/{filename}", 1)
