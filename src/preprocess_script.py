#!/usr/bin/env python3
from core.headPosition import detectFace2
from core.CameraUtils import get_offsets
import cv2
import glob
import numpy as np

from PIL import Image

# for image in glob.glob("./src/data/datasets/Selfie-dataset/images/*.jpg"):
#     img = Image.open(image)
#     img = np.asarray(img)
#     # img = img[530:1030, 250:750]
#     img = cv2.resize(img, (640, 360))
#     detected = detectFace2(img)
#     # print(f'{image} - {detected}')
#     if len(detected.detections) > 0:
#         print(f'{image}: {detected.detections[0].bounding_box}')
#         
#         x1 = detected.detections[0].bounding_box.origin_x
#         x2 = x1 + detected.detections[0].bounding_box.width
#         y1 = detected.detections[0].bounding_box.origin_y
#         y2 = y1 + detected.detections[0].bounding_box.height        
#         print(f'Input box: {x1}, {x2}, {y1}, {y2} = {x2-x1}:{y2-y1}')
#
#         x1, x2, y1, y2 = get_offsets(x1, x2, y1, y2)
#         print(f'Corrected size: {x1}, {x2}, {y1}, {y2} = {x2-x1}:{y2-y1}')
#
#         img = img[y1:y2, x1:x2]
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         cv2.imwrite(f"{image[:-4]}_corrected.png", img)
#
#     else:
#         print(f"{image}: Not found")
#     # cv2.imwrite(f'{image[:-4]}_resized.png', img)

for image in glob.glob("./src/data/datasets/prepared/*.png"):
    img = Image.open(image)
    img = np.asarray(img)

    if img.shape != (224, 224, 3):
        print(f'{image}: Shape incorrect. Current shape is: {img.shape}')
