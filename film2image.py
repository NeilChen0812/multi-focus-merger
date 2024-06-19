import os
import shutil
import cv2


def film2image(video_path, output_folder="output", sampling_interval=1):
    if not isinstance(sampling_interval, int):
        raise ValueError("Sampling interval must be an integer")
    if not isinstance(output_folder, str):
        raise ValueError("Output folder must be a string")
    if not os.path.exists(video_path):
        raise FileNotFoundError("Video file not found: {}".format(video_path))

    if os.path.exists(output_folder):
        print("Delete old result in folder: {}".format(output_folder))
        shutil.rmtree(output_folder)
    else:
        print("Create new folder: {}".format(output_folder))
    os.makedirs(output_folder)

    vc = cv2.VideoCapture(video_path)
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx in range(0, frame_count, sampling_interval):
        vc.set(1, idx)
        ret, frame = vc.read()

        if frame is not None:
            file_name = '{}\{:04d}.jpg'.format(output_folder, idx)
            cv2.imwrite(file_name, frame)

        print("\rProcess: {}/{}".format(idx+1, frame_count), end='')
    vc.release()
    print("  (Done)")


if __name__ == '__main__':
    video_path = 'videos\P1500667.MP4'
    film2image(video_path, "test", 10)
