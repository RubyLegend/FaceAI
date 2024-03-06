#!/usr/bin/env python3
from core.headPosition import detectFace2
from core.network import get_offsets
import cv2
import glob
import numpy as np

from PIL import Image

print("Imports done")

# bbox = {}
# faces = {}
faces_train = {}
faces_test = {}
faces_valid = {}
#
# train = []
# test = []
# valid = []
#
# with open("./src/data/datasets/list_eval_partition.txt") as file:
#     for line in file:
#         filename, partition = line.split(' ')
#         # print(f'{filename}, {partition}')
#         if int(partition) == 0:
#             train.append(filename)
#         elif int(partition) == 1:
#             test.append(filename)
#         else:
#             valid.append(filename)

# with open("./src/data/datasets/list_bbox_celeba.txt") as file:
#     total_size = int(file.readline())
#     print(f'Total count of images: {total_size}')
#     headers = file.readline()
#     for line in file:
#         filename, x, y, width, height = line.split()
#         print(f'{filename}, {x}, {y}, {width}, {height}')
#         x, y, width, height = int(x), int(y), int(width), int(height)
#         print(f'{filename}, {x}, {y}, {width}, {height}')
#         img = Image.open(f"./src/data/datasets/img_align_celeba/{filename}")
#         img = np.asarray(img)
#         print(img.shape)
#         # img = img[y:y+100, x:x+width]
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         cv2.imshow("img", img)
#         cv2.waitKey()

file2 = open("./src/data/datasets/list_eval_partition.txt")

with open("./src/data/datasets/identity_CelebA.txt") as file:
    for line in file:
        filename2, partition = file2.readline().split()
        partition = int(partition)
        filename, identity = line.split()
        identity = int(identity)
        if partition == 0:
            if identity not in faces_train:
                faces_train[identity] = []
            faces_train[identity].append(filename)
        elif partition == 1:
            if identity not in faces_test:
                faces_test[identity] = []
            faces_test[identity].append(filename)
        else:
            if identity not in faces_valid:
                faces_valid[identity] = []
            faces_valid[identity].append(filename)

file2.close()

i = np.random.choice(list(faces_train.keys()))
for filename in faces_train[i]:
    img = Image.open(f'./src/data/datasets/img_align_celeba/{filename}')
    img = np.asarray(img)
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("img", img)
    cv2.waitKey()

# for image in glob.glob("./src/data/datasets/RGBD_Face_dataset_training/*.png"):
#     img = Image.open(image)
#     img = np.asarray(img)
#     # img = cv2.imread(image)
#     # img = img[400:1300, 200:800]
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
#         x1, x2, y1, y2, is_resize = get_offsets(x1, x2, y1, y2)
#         print(f'Corrected size: {x1}, {x2}, {y1}, {y2} = {x2-x1}:{y2-y1}')
#
#         img = img[y1:y2, x1:x2]
#         if is_resize:
#             img = cv2.resize(img, (224, 224))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         cv2.imshow('img', img)
#         cv2.waitKey()
#
#         # cv2.imwrite(f"{image[:-4]}_corrected.png", img)
#
#     else:
#         print(f"{image}: Not found")
#     # cv2.imwrite(f'{image[:-4]}_resized.png', img)

# for image in glob.glob("./src/data/datasets/prepared/*.png"):
#     img = Image.open(image)
#     img = np.asarray(img)
#
#     if img.shape != (224, 224, 3):
#         print(f'{image}: Shape incorrect. Current shape is: {img.shape}')

# face_cascade = cv2.CascadeClassifier("./src/haarcascade_frontalface_default.xml")
#
# for image in glob.glob("./src/data/datasets/RGBD_Face_dataset_training/*.png"):
#     img = Image.open(image)
#     img = np.asarray(img)
#     # img = cv2.imread(image)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     # Display the output
#     cv2.imshow('img', img)
#     cv2.waitKey()
#
