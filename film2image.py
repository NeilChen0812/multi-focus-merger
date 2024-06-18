import os
import shutil
import cv2

video_path = './videos/P1500677.MP4'
output_folder = './temp-images'

# Delete old result folder
if os.path.exists(output_folder):
    print("Delete old result folder: {}".format(output_folder))
    shutil.rmtree(output_folder)
print("create folder: {}".format(output_folder))
os.makedirs(output_folder)

vc = cv2.VideoCapture(video_path)
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)
sampling_interval = 10

for idx in range(0, frame_count, sampling_interval):
    vc.set(1, idx)
    ret, frame = vc.read()

    if frame is not None:
        file_name = '{}\{:04d}.jpg'.format(output_folder, idx)
        cv2.imwrite(file_name, frame)

    print("\rprocess: {}/{}".format(idx+1, frame_count), end='')
vc.release()
