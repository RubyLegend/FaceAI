#!/usr/bin/env python3
from email.mime import image
import os, inspect
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Dense,
    Activation,
    Flatten,
    Dropout,
    Lambda,
    ELU,
    GlobalAveragePooling2D,
    Input,
    BatchNormalization,
    SeparableConv2D,
    Subtract,
    concatenate,
    Embedding,
    Reshape,
)
from tensorflow.keras.activations import relu, softmax  # type: ignore
from tensorflow.keras.layers import Conv2D as Convolution2D  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, AdamW  # type: ignore
from tensorflow.keras.optimizers.schedules import (  # type: ignore
    ExponentialDecay,
    InverseTimeDecay,
)  # type:ignore
from tensorflow.keras.losses import (  # type: ignore
    SparseCategoricalCrossentropy as SCC,
    MeanAbsoluteError,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory  # type: ignore
from tensorflow.image import (  # type: ignore
    random_flip_left_right,
    random_flip_up_down,
    random_brightness,
    random_contrast,
    random_saturation,
)

# from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K  # type: ignore
from tensorflow.keras.utils import plot_model, split_dataset  # type: ignore
from tensorflow.keras.applications import mobilenet_v2  # type: ignore
import tensorflow as tf
import keras

from scipy import ndimage

import numpy as np
import glob
import random
import cv2
from matplotlib import pyplot as plt
import pickle
from PIL import Image
from typing import Tuple, Dict

import sys

if sys.argv[0] == "./antispoof.py":
    from headPosition import detectFace2
else:
    from core.headPosition import detectFace2


def initGPU():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3500)]
        )

    logical_gpus = tf.config.list_logical_devices("GPU")
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")


INPUT_WIDTH_HEIGHT = 224

# Neural network structure


def getNetworkModel():
    img_input = Input(shape=(INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, 3))

    x = img_input

    # Neural network structure
    x = Convolution2D(32, (7, 7), strides=1, padding="valid")(x)
    x = Convolution2D(32, (7, 7), strides=1, padding="valid")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

    x = Convolution2D(48, (6, 6), strides=1, padding="valid")(x)
    x = Convolution2D(48, (6, 6), strides=1, padding="valid")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

    x = Convolution2D(64, (4, 4), strides=1, padding="valid")(x)
    x = Convolution2D(64, (4, 4), strides=1, padding="valid")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

    x = Convolution2D(128, (3, 3), strides=1, padding="valid")(x)
    x = Convolution2D(128, (3, 3), strides=1, padding="valid")(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.1)(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

    x = Flatten()(x)

    x = Dense(2 * INPUT_WIDTH_HEIGHT, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(INPUT_WIDTH_HEIGHT, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation="sigmoid")(x)
    # End neural network

    out = x

    model = Model(img_input, out)

    # Optimizer
    lr_schedule = ExponentialDecay(1e-5, decay_steps=382, decay_rate=1e-6)
    adam = Adam(learning_rate=1e-10)
    adamw = AdamW()
    rms = RMSprop()

    sgd = SGD(learning_rate=0.00006, momentum=0.9)

    model.compile(
        optimizer=sgd,
        metrics=[
            "accuracy",
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.F1Score(average="weighted", threshold=0.4),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            custom_f1,
        ],
        loss="binary_crossentropy",
        # f1_score
        # presicion_recall
    )
    plot_model(model, show_shapes=True, expand_nested=True, to_file="plot.png")
    model.summary()

    return model


@keras.saving.register_keras_serializable()
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, **kwargs):
        super(ResnetIdentityBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        filters1, filters2, filters3 = self.filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def get_config(self):
        config = super(ResnetIdentityBlock, self).get_config().copy()
        config.update({"kernel_size": self.kernel_size, "filters": self.filters})
        return config

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


@keras.saving.register_keras_serializable()
class ResidualBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        filters1, filters2 = self.filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, kernel_size, padding="same")
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
        self.bn2b = tf.keras.layers.BatchNormalization()

    def get_config(self):
        config = super(ResidualBlock, self).get_config().copy()
        config.update({"kernel_size": self.kernel_size, "filters": self.filters})
        return config

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


def getResNetModel():
    img_input = Input(shape=(INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, 3))

    x = img_input

    # Neural network structure

    x = Convolution2D(64, (7, 7), strides=2, padding="same")(img_input)
    x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

    x = ResnetIdentityBlock((3, 3), (4, 4, 64))(x)
    x = ResnetIdentityBlock((3, 3), (4, 4, 64))(x)

    x = Convolution2D(128, (3, 3), strides=2, padding="same")(x)

    x = ResnetIdentityBlock((3, 3), (4, 4, 128))(x)
    x = ResnetIdentityBlock((3, 3), (4, 4, 128))(x)

    x = Convolution2D(256, (3, 3), strides=2, padding="same")(x)

    x = ResnetIdentityBlock((3, 3), (4, 4, 256))(x)
    x = ResnetIdentityBlock((3, 3), (4, 4, 256))(x)

    x = Convolution2D(512, (3, 3), strides=2, padding="same")(x)

    x = ResnetIdentityBlock((3, 3), (4, 4, 512))(x)
    x = ResnetIdentityBlock((3, 3), (4, 4, 512))(x)

    x = GlobalAveragePooling2D()(x)

    # x = Dense(2 * INPUT_WIDTH_HEIGHT, activation="relu")(x)
    # x = BatchNormalization()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    # x = Dense(128, activation="relu")(x)
    # x = BatchNormalization()(x)

    x = Dense(1, activation="sigmoid")(x)
    # End neural network

    out = x

    model = Model(img_input, out)

    # Optimizer
    lr_schedule = ExponentialDecay(0.01, decay_steps=3000, decay_rate=0.1)
    adam = Adam(learning_rate=1e-10)
    adamw = AdamW()
    rms = RMSprop()

    sgd = SGD(learning_rate=lr_schedule, momentum=0.9)

    model.compile(
        optimizer=sgd,
        metrics=[
            # "accuracy",
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.F1Score(average="weighted", threshold=0.4),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            custom_f1,
        ],
        loss="binary_crossentropy",
        # f1_score
        # presicion_recall
    )
    plot_model(model, show_shapes=True, expand_nested=True, to_file="plot.png")
    model.summary()

    return model


def getResNetModel2():
    img_input = Input(shape=(INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, 3))

    x = img_input

    # Neural network structure

    x = Convolution2D(64, (7, 7), strides=2, padding="same")(img_input)
    x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

    x = ResidualBlock((3, 3), (64, 64))(x)
    x = ResidualBlock((3, 3), (64, 64))(x)

    x = Convolution2D(128, (3, 3), strides=2, padding="same")(x)

    x = ResidualBlock((3, 3), (128, 128))(x)
    x = ResidualBlock((3, 3), (128, 128))(x)

    x = Convolution2D(256, (3, 3), strides=2, padding="same")(x)

    x = ResidualBlock((3, 3), (256, 256))(x)
    x = ResidualBlock((3, 3), (256, 256))(x)

    x = Convolution2D(512, (3, 3), strides=2, padding="same")(x)

    x = ResidualBlock((3, 3), (512, 512))(x)
    x = ResidualBlock((3, 3), (512, 512))(x)

    x = GlobalAveragePooling2D()(x)

    # x = Dense(2 * INPUT_WIDTH_HEIGHT, activation="relu")(x)
    # x = BatchNormalization()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    # x = Dense(128, activation="relu")(x)
    # x = BatchNormalization()(x)

    x = Dense(1, activation="sigmoid")(x)
    # End neural network

    out = x

    model = Model(img_input, out)

    # Optimizer
    lr_schedule = ExponentialDecay(0.01, decay_steps=3000, decay_rate=0.1)
    adam = Adam(learning_rate=1e-10)
    adamw = AdamW()
    rms = RMSprop()

    sgd = SGD(learning_rate=lr_schedule, momentum=0.9)

    model.compile(
        optimizer=sgd,
        metrics=[
            # "accuracy",
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.F1Score(average="weighted", threshold=0.4),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            custom_f1,
        ],
        loss="binary_crossentropy",
        # f1_score
        # presicion_recall
    )
    plot_model(model, show_shapes=True, expand_nested=True, to_file="plot.png")
    model.summary()

    return model


def loadAntispoofModel(path="./faceantispoof_network.keras", compile=False):
    model = load_model(
        path, compile=False, custom_objects={"ResnetIdentityBlock": ResnetIdentityBlock}
    )
    model.summary()

    if compile:
        lr_schedule = ExponentialDecay(0.01, decay_steps=3000, decay_rate=0.1)
        sgd = SGD(learning_rate=lr_schedule, momentum=0.9)  # 0.0001
        model.compile(
            optimizer=sgd,
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.F1Score(average="weighted", threshold=0.4),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

    return model


def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def load_imagenet_mobilenetv2():
    base_model = mobilenet_v2.MobileNetV2(
        weights="imagenet",
        input_shape=(
            INPUT_WIDTH_HEIGHT,
            INPUT_WIDTH_HEIGHT,
            3,
        ),  # already imagenet input size
        include_top=False,  # Just removes top layer
        classes=1000,  # If using imagenet and include_top=True, classes should be 1000 (default)
        classifier_activation="softmax",  # sigmoid is not allowed, available: None, softmax
    )
    base_model.trainable = True  # Locking network weights, so that is is not modifiable
    # Loading speed pretty fast, to say the least

    # Building another network around base model
    input = Input((INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, 3))
    x = input

    x = base_model(
        input, training=True
    )  # Just for the sake of protection, locking training here as well

    x = GlobalAveragePooling2D()(x)
    x = Activation("relu")(x)

    # x = Flatten()(x)

    # x = Dense(2 * INPUT_WIDTH_HEIGHT, activation="relu")(x)
    # x = BatchNormalization()(x)

    x = Dense(INPUT_WIDTH_HEIGHT, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation="sigmoid")(x)

    # End neural network

    out = x

    out_model = tf.keras.Model(input, out)

    out_model.compile(
        optimizer=SGD(1e-4),  # Very low learning rate
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.F1Score(average="weighted", threshold=0.4),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            custom_f1,
        ],
    )

    plot_model(out_model, show_shapes=True, expand_nested=True, to_file="plot.png")
    out_model.summary()

    return out_model


def resize_image_with_pixel_change(image, change_x, change_y, order=0):
    """
    Resizes a 2D image array based on desired pixel changes in each dimension.

    Args:
        image: The original image as a 2D NumPy array.
        change_x: Number of pixels to decrease/increase in the x-dimension.
        change_y: Number of pixels to decrease/increase in the y-dimension.
        order: Interpolation order (0 - nearest neighbor). Defaults to 0.

    Returns:
        A new NumPy array representing the resized image.
    """

    # Calculate scaling factors
    original_rows, original_cols, _ = image.shape
    scale_x = (original_cols + change_x) / original_cols
    scale_y = (original_rows + change_y) / original_rows

    # Resize using zoom with nearest neighbor interpolation
    resized_image = ndimage.zoom(image, (scale_x, scale_y, 1), order=order)
    return resized_image


def crop_upscale(image, target_x, target_y, detection_result, debug=False):
    # print(detection_result)
    new_image = image[
        detection_result.origin_y : detection_result.origin_y + detection_result.width,
        detection_result.origin_x : detection_result.origin_x + detection_result.height,
    ]
    new_image = np.array(new_image)
    new_image2 = resize_image_with_pixel_change(
        new_image, target_x - detection_result.width, target_y - detection_result.height
    )
    if new_image2.shape != (160, 160, 3):
        if debug:
            print("upscale shape mismatch")
            print(f"input image shape: {image.shape}")
            print(
                f"upscale debug: {detection_result.origin_x}:{detection_result.origin_x+detection_result.width}"
            )
            print(
                f"upscale debug: {detection_result.origin_y}:{detection_result.origin_y+detection_result.height}"
            )
            print(
                f"upscale debug: {target_x - detection_result.width}, {target_y - detection_result.height}"
            )
            print(
                f"input was: {new_image.shape}, {target_x}, {target_y}, {detection_result}"
            )
            print(f"new_image shape: {new_image2.shape}")
        return None
    else:
        return new_image2


def crop_downscale(image, target_x, target_y, detection_result, debug=False):
    # print(detection_result)
    new_image = image[
        detection_result.origin_y : detection_result.origin_y + detection_result.width,
        detection_result.origin_x : detection_result.origin_x + detection_result.height,
    ]
    new_image = np.array(new_image)
    new_image2 = resize_image_with_pixel_change(
        new_image, target_x - detection_result.width, target_y - detection_result.height
    )
    if new_image2.shape != (160, 160, 3):
        if debug:
            print("downscale shape mismatch")
            print(f"input image shape: {image.shape}")
            print(
                f"downscale debug: {detection_result.origin_x}:{detection_result.origin_x+detection_result.width}"
            )
            print(
                f"downscale debug: {detection_result.origin_y}:{detection_result.origin_y+detection_result.height}"
            )
            print(
                f"downscale debug: {target_x - detection_result.width}, {target_y - detection_result.height}"
            )
            print(
                f"input was: {new_image.shape}, {target_x}, {target_y}, {detection_result}"
            )
            print(f"new_image shape: {new_image2.shape}")
        return None
    else:
        return new_image2


def generateData(train=False):
    faces_data = []
    faces_data_labels = []
    if train:
        with open(
            "/home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/NUAA/client_train_raw.txt",
            "r",
        ) as file:
            for line in file:
                img_path = line[:-1]
                # os.system(f"cp {img_path} /home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/NUAA/train/1/.")
                img = Image.open(img_path)
                img = np.asarray(img)
                result = detectFace2(img)
                if len(result.detections) != 0:
                    bbox = result.detections[0].bounding_box
                    if bbox.width < INPUT_WIDTH_HEIGHT:
                        img = crop_upscale(
                            img, INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, bbox
                        )
                    elif bbox.width > INPUT_WIDTH_HEIGHT:
                        img = crop_downscale(
                            img, INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, bbox
                        )
                    else:
                        img = img[
                            bbox.origin_y : bbox.origin_y + bbox.width,
                            bbox.origin_x : bbox.origin_x + bbox.height,
                        ]

                    if type(img) != type(None):
                        if img.shape != (160, 160, 3):
                            print("shape mismatch at insert client train")
                            if bbox.width < INPUT_WIDTH_HEIGHT:
                                print("upscaling")
                                print(f"Img shape: {img.shape}")
                            else:
                                print("downscaling")
                                print(f"Img shape: {img.shape}")
                        faces_data.append(img)
                        faces_data_labels.append(1.0)
                    else:
                        print("Image skipped")

        with open(
            "/home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/NUAA/imposter_train_raw.txt",
            "r",
        ) as file:
            for line in file:
                img_path = line[:-1]
                # os.system(f"cp {img_path} /home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/NUAA/train/0/.")
                img = Image.open(img_path)
                img = np.asarray(img)
                result = detectFace2(img)
                if len(result.detections) != 0:
                    bbox = result.detections[0].bounding_box
                    if bbox.width < INPUT_WIDTH_HEIGHT:
                        img = crop_upscale(
                            img, INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, bbox
                        )
                    elif bbox.width > INPUT_WIDTH_HEIGHT:
                        img = crop_downscale(
                            img, INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, bbox
                        )
                    else:
                        img = img[
                            bbox.origin_y : bbox.origin_y + bbox.width,
                            bbox.origin_x : bbox.origin_x + bbox.height,
                        ]

                    if type(img) != type(None):
                        if img.shape != (160, 160, 3):
                            print("shape mismatch at insert imposter train")
                            if bbox.width < INPUT_WIDTH_HEIGHT:
                                print("upscaling")
                                print(f"Img shape: {img.shape}")
                            else:
                                print("downscaling")
                                print(f"Img shape: {img.shape}")
                        faces_data.append(img)
                        faces_data_labels.append(0.0)
                    else:
                        print("Image skipped")
    else:
        with open(
            "/home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/NUAA/client_test_raw.txt",
            "r",
        ) as file:
            for line in file:
                img_path = line[:-1]
                # os.system(f"cp {img_path} /home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/NUAA/valid/1/.")
                img = Image.open(img_path)
                img = np.asarray(img)
                result = detectFace2(img)
                if len(result.detections) != 0:
                    bbox = result.detections[0].bounding_box
                    if bbox.width < INPUT_WIDTH_HEIGHT:
                        img = crop_upscale(
                            img, INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, bbox
                        )
                    elif bbox.width > INPUT_WIDTH_HEIGHT:
                        img = crop_downscale(
                            img, INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, bbox
                        )
                    else:
                        img = img[
                            bbox.origin_y : bbox.origin_y + bbox.width,
                            bbox.origin_x : bbox.origin_x + bbox.height,
                        ]

                    if type(img) != type(None):
                        if img.shape != (160, 160, 3):
                            print("shape mismatch at insert client test")
                            if bbox.width < INPUT_WIDTH_HEIGHT:
                                print("upscaling")
                                print(f"Img shape: {img.shape}")
                            else:
                                print("downscaling")
                                print(f"Img shape: {img.shape}")
                        faces_data.append(img)
                        faces_data_labels.append(1.0)
                    else:
                        print("Image skipped")

        with open(
            "/home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/NUAA/imposter_test_raw.txt",
            "r",
        ) as file:
            for line in file:
                img_path = line[:-1]
                # os.system(f"cp {img_path} /home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/NUAA/valid/0/.")
                img = Image.open(img_path)
                img = np.asarray(img)
                result = detectFace2(img)
                if len(result.detections) != 0:
                    bbox = result.detections[0].bounding_box
                    if bbox.width < INPUT_WIDTH_HEIGHT:
                        img = crop_upscale(
                            img, INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, bbox
                        )
                    elif bbox.width > INPUT_WIDTH_HEIGHT:
                        img = crop_downscale(
                            img, INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT, bbox
                        )
                    else:
                        img = img[
                            bbox.origin_y : bbox.origin_y + bbox.width,
                            bbox.origin_x : bbox.origin_x + bbox.height,
                        ]

                    if type(img) != type(None):
                        if img.shape != (160, 160, 3):
                            print("shape mismatch at insert imposter test")
                            if bbox.width < INPUT_WIDTH_HEIGHT:
                                print("upscaling")
                                print(f"Img shape: {img.shape}")
                            else:
                                print("downscaling")
                                print(f"Img shape: {img.shape}")
                        faces_data.append(img)
                        faces_data_labels.append(0.0)
                    else:
                        print("Image skipped")

    faces_data = np.asarray(faces_data)
    faces_data_labels = np.asarray(faces_data_labels)
    return faces_data, faces_data_labels


def train(model):
    # full_train_data, full_train_data_labels = generateData(True)
    # full_test_data, full_test_data_labels = generateData()
    # data_generator = ImageDataGenerator()

    """
    training_dataset = data_generator.flow_from_directory(
        directory="/home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/face-anti-spoof/train",
        batch_size=4,
        target_size=(INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT),
        color_mode="rgb",
        class_mode="binary",
    )
    validation_dataset = data_generator.flow_from_directory(
        directory="/home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/face-anti-spoof/test",
        batch_size=4,
        target_size=(INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT),
        color_mode="rgb",
        class_mode="binary",
    )
    """

    training_dataset = image_dataset_from_directory(
        directory="/home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/face-anti-spoof-join/train",
        batch_size=64,
        image_size=(INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT),
        color_mode="rgb",
        label_mode="binary",
        verbose=True,
    )

    validation_dataset = image_dataset_from_directory(
        directory="/home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/face-anti-spoof-join/test",
        batch_size=64,
        image_size=(INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT),
        color_mode="rgb",
        label_mode="binary",
        verbose=True,
    )

    print(f"Training as: {training_dataset.class_names}")

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=5,
        verbose=1,
        mode="min",
        restore_best_weights=True,
        start_from_epoch=20,
    )
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath="./faceantispoof_network.keras", save_best_only=True
    )

    new_model = model.fit(
        # data,
        # full_train_data, full_train_data_labels,
        training_dataset,
        # steps_per_epoch=100,
        epochs=150,
        # validation_data=(full_test_data, full_test_data_labels),
        validation_data=validation_dataset,
        # validation_steps=1,
        callbacks=[
            checkpointer,
            early_stop,
        ],
    )
    model.save("faceantispoof_network.keras")

    plt.plot(new_model.history["binary_accuracy"])
    plt.plot(new_model.history["val_binary_accuracy"])

    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    plt.savefig("history.png")

    plt.clf()

    plt.plot(new_model.history["loss"])
    plt.plot(new_model.history["val_loss"])

    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    plt.savefig("history_loss.png")

    plt.clf()

    plt.plot(new_model.history["f1_score"])
    plt.plot(new_model.history["val_f1_score"])

    plt.title("model f1")
    plt.ylabel("f1score")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    plt.savefig("history_fbeta_score.png")

    plt.clf()

    plt.plot(new_model.history["custom_f1"])
    plt.plot(new_model.history["val_custom_f1"])

    plt.title("model f1")
    plt.ylabel("f1score")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    plt.savefig("history_custom_f1.png")

    plt.clf()

    plt.plot(new_model.history["precision"])
    plt.plot(new_model.history["val_precision"])

    plt.title("model precision")
    plt.ylabel("precision")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    plt.savefig("history_precision.png")

    plt.clf()

    plt.plot(new_model.history["recall"])
    plt.plot(new_model.history["val_recall"])

    plt.title("model recall")
    plt.ylabel("recall")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    plt.savefig("history_recall.png")


def test(model):
    from os import listdir
    from os.path import isfile, join

    validation_dataset = image_dataset_from_directory(
        directory="/home/ruby/Code_Workspace/ITvR/CourseWork/src/data/datasets/face-anti-spoof-join/test",  # val
        batch_size=1,
        image_size=(INPUT_WIDTH_HEIGHT, INPUT_WIDTH_HEIGHT),
        color_mode="rgb",
        label_mode="binary",
        verbose=True,
    )

    # lr_schedule = ExponentialDecay(0.01, decay_steps=10000, decay_rate=0.1)
    # sgd = SGD(learning_rate=lr_schedule, momentum=0.9)

    # model.compile(
    #     optimizer=sgd,
    #     metrics=[
    #         # "accuracy",
    #         tf.keras.metrics.BinaryAccuracy(),
    #         tf.keras.metrics.F1Score(average="weighted", threshold=0.4),
    #         tf.keras.metrics.Precision(),
    #         tf.keras.metrics.Recall(),
    #         custom_f1,
    #     ],
    #     loss="binary_crossentropy",
    #     # f1_score
    #     # presicion_recall
    # )
    results = model.evaluate(validation_dataset)
    print(f"Evaluation results: {results}")

    mypath = "/home/ruby/Code_Workspace/ITvR/CourseWork/src/data/"
    files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in files:
        frame = Image.open(file)
        frame = np.array(frame)
        frame = cv2.resize(frame, (224, 224))
        # cv2.imwrite("/tmp/img.jpg", frame)

        res = model.predict(np.asarray([frame]))
        print(f"Prediction result for {file}: {res}")


def testModel(model, frame):
    frame = cv2.resize(frame, (224, 224))

    res = model.predict(np.asarray([frame]))
    # print(f"Prediction result: {res}")
    if res <= 0.4:
        return True, res
    return False, res


if __name__ == "__main__":
    initGPU()
    # If you want new blank network - uncomment line below
    # model = getResNetModel()
    # model = load_imagenet_mobilenetv2()

    # import visualkeras

    # visualkeras.layered_view(model, to_file="modelview_antispoof.png")

    # If you want to resume training, but with another dataset - uncomment line below
    model = loadAntispoofModel(compile=True)

    # train(model)
    test(model)
