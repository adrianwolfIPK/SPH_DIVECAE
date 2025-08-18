import cv2
import os
import re

image_folder = input("Directory containing your PNGs: ") # folder with images
title = "movie"
output_video = os.path.join(image_folder, f"{title}.mp4") # destination file (mp4)
fps = 24

pattern = re.compile(r"(.*)\.(\d+)\.png")
images = []

for filename in os.listdir(image_folder):
    match = pattern.match(filename)
    if match:
        index = int(match.group(2))
        images.append((index, filename))

images.sort(key=lambda x: x[0])
image_files = [os.path.join(image_folder, f[1]) for f in images]

if not image_files:
    raise RuntimeError("No matching .png files found!")

frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for image_path in image_files:
    frame = cv2.imread(image_path)
    if frame is not None:
        video.write(frame)

video.release()
print(f"Video created: {output_video}")
