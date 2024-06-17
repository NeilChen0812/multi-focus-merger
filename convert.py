import os
import cv2
import glob

video_path = "P1500704.MP4"
output_folder = '02'

if os.path.isdir(output_folder):
    print("Delete old result folder: {}".format(output_folder))
    os.system("rm -rf {}".format(output_folder))
os.system("mkdir {}".format(output_folder))
print("create folder: {}".format(output_folder))

vc = cv2.VideoCapture(video_path)
fps = vc.get(cv2.CAP_PROP_FPS)
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)
video = []

for idx in range(frame_count):
    vc.set(1, idx)
    ret, frame = vc.read()

    if frame is not None:
        file_name = '{}\{:04d}.jpg'.format(output_folder, idx)
        cv2.imwrite(file_name, frame)

    print("\rprocess: {}/{}".format(idx+1, frame_count), end='')
vc.release()
