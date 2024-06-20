# Code is blatantly stolen from github.
# I'll learn it a little bit later
# and probably rewrite it
# because I want to have full understanding of what is happening here
# and most importantly, how and why.
import sys
from tarfile import fully_trusted_filter

import tensorflow as tf

# from tensorflow import keras

from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import (
    Dense,
    Activation,
    Flatten,
    Dropout,
    Lambda,
    ELU,
    LeakyReLU,
    GlobalAveragePooling2D,
    Input,
    BatchNormalization,
    SeparableConv2D,
    Subtract,
    concatenate,
)
from keras.activations import relu, softmax
from keras.layers import Conv2D as Convolution2D
from keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    Concatenate,
)
from keras.optimizers import Adam, SGD, RMSprop, AdamW, Adagrad, Adamax
from keras.optimizers.schedules import ExponentialDecay, InverseTimeDecay
from keras.losses import (
    SparseCategoricalCrossentropy as SCC,
    MeanAbsoluteError,
)

# from tensorflow.keras.regularizers import l2
# from tensorflow.keras import backend as K
from keras.ops import square, sqrt, mean, sum, maximum, equal, cast
from keras.backend import epsilon
from keras.utils import plot_model, split_dataset
import keras
import imgaug as ia
import imgaug.augmenters as iaa

from sklearn.utils import shuffle

import numpy as np
import glob
import random
import cv2
from matplotlib import pyplot as plt
import pickle
from PIL import Image
import itertools


from core.headPosition import detectFace

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3000)]
    )

logical_gpus = tf.config.list_logical_devices("GPU")
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

# prepared_images_all = glob.glob(f"src/data/datasets/prepared/*.png")
# prep_left, prep_right = split_dataset(np.array(prepared_images_all), left_size=0.7)
# prep_left = [val.numpy() for val in prep_left]
# prep_right = [val.numpy() for val in prep_right]

faces_train = {}
faces_test = {}
faces_valid = {}
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


def refresh_user_pictures():
    # train - 70%
    # test - 20%
    # valid - 10%
    images = [val for val in range(0, 36)]
    images = random.sample(images, len(images))
    train_end = int(len(images) * 0.7)
    test_end = int(len(images) * 0.9)
    valid_end = len(images)

    faces_train["myface"] = [f"{it}.jpg" for it in range(0, train_end)]
    faces_test["myface"] = [f"{it}.jpg" for it in range(train_end, test_end)]
    faces_valid["myface"] = [f"{it}.jpg" for it in range(test_end, len(images))]


# keys_train = list(faces_train.keys())
# keys_test = list(faces_test.keys())
# keys_valid = list(faces_valid.keys())


# prepared_images = {}
# for i in range(1, 11):
#     left, right = split_dataset(np.array(glob.glob(f"src/data/datasets/prepared/{i:03d}_*.png")), left_size = 0.7)
#     prepared_images[i] = {}
#     prepared_images[i][0] = [val.numpy() for val in left]
#     prepared_images[i][1] = [val.numpy() for val in right]


@keras.saving.register_keras_serializable()
def euclidean_distance(inputs):
    assert len(inputs) == 2, "Euclidean distance needs 2 inputs, %d given" % len(inputs)
    u, v = inputs
    # sum_square = K.sum(K.square(u - v), axis=1, keepdims=True)
    sqr = square(u - v)
    return sqrt(maximum(sqr, epsilon()))


@keras.saving.register_keras_serializable()
def treshold_function(input):
    return cast(input >= 0.8, "int16")


@keras.saving.register_keras_serializable()
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    assert shape1[0] == shape2[0], "Shapes of two images doesn't match"
    return (shape1[0], 64)


@keras.saving.register_keras_serializable()
def treshold_shape(shape):
    return (shape[0], 1)


def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


@keras.saving.register_keras_serializable()
def accuracy(y_true, y_pred):
    return mean(equal(y_true, cast(y_pred >= 0.8, y_true.dtype)))


@keras.saving.register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = square(y_pred)
    margin_square = square(maximum(margin - y_pred, epsilon()))
    # return K.mean(
    #     (1.0 - y_true) * 0.5 * K.square(y_pred)
    #     + y_true * 0.5 * K.square(K.maximum(margin - y_pred, 0.0))
    # )
    # return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return mean((1.0 - y_true) * 0.5 * square_pred + y_true * 0.5 * margin_square)


# return K.mean( K.square(y_pred) )
# --------------
@keras.saving.register_keras_serializable()
def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1, 1), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    left = Convolution2D(expand, (1, 1), padding="valid")(x)
    left = Activation("relu")(left)

    right = Convolution2D(expand, (3, 3), padding="same")(x)
    right = Activation("relu")(right)

    x = Concatenate(axis=3)([left, right])
    # x = tf.concat([left, right], 1)
    return x


# class RandomInvert(tf.keras.layers.Layer):
#     def __init__(self, max_value = float(255.0), factor=0.5, **kwargs):
#         super().__init__(**kwargs)
#         self.factor = factor
#         self.max_value = max_value
#
#     def call(self, x):
#         if tf.random.uniform([]) < self.factor:
#             x = (self.max_value - x)
#         return x
#
# data_augmentation = tf.keras.Sequential([
#     RandomInvert(max_value = 224),
#     tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#     tf.keras.layers.RandomRotation((-0.4, 0.4)),
#     tf.keras.layers.RandomBrightness(factor=(-0.2, 0.2), value_range=(0., 1.)),
#     tf.keras.layers.GaussianNoise(0.005),
#     tf.keras.layers.RandomZoom(height_factor=(-0.4, 0.4)),
#     tf.keras.layers.RandomContrast(factor=(0.1, 0.9)),
#     tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
# ])
#
# data_augmentation.compile(optimizer=Adam(), loss=SCC())

# vflip = iaa.Flipud(p=1.0)
# hflip = iaa.Fliplr(p=1.0)
# crop1 = iaa.Crop(percent=(0, 0.15))
# noise = iaa.AdditiveGaussianNoise(10, 40)
# shear = iaa.Affine(shear=(-40, 40))
# contrast = iaa.GammaContrast((0.5, 2.0))
# contrast_sig = iaa.SigmoidContrast(gain=(5, 10), cutoff=(0.4, 0.6))
# contrast_lin = iaa.LinearContrast((0.6, 0.4))

# augments = [vflip, hflip, crop1, noise, shear, contrast]


# def data_augmentation(img):
#     for aug in augments:
#         img = aug.augment_image(img)

#     return img


def getNewModel2():

    # SqueezeNet architecture

    img_input = Input(shape=(224, 224, 3))  # was 200, 200, 4

    x = Convolution2D(64, (7, 7), strides=2, padding="same")(img_input)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = BatchNormalization()(x)

    x = fire(x, squeeze=16, expand=64)

    x = fire(x, squeeze=16, expand=64)

    x = fire(x, squeeze=32, expand=128)

    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Activation("relu")(x)

    x = fire(x, squeeze=32, expand=128)

    x = fire(x, squeeze=48, expand=192)

    x = fire(x, squeeze=48, expand=192)

    x = fire(x, squeeze=64, expand=256)

    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Activation("relu")(x)

    x = fire(x, squeeze=64, expand=256)

    x = Dropout(0.5)(x)  # Only if learning the SqueezeNet itself

    x = Convolution2D(1000, (1, 1), strides=1)(x)
    # x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(
        x
    )  # I believe this layer is obsolete for a siamese network
    # # Because there will be calculated eu

    x = Dense(64, activation="relu")(x)

    out = x

    modelsqueeze = Model(img_input, out)

    modelsqueeze.summary()

    # SqueezeNet end

    # # # Second neural model - subnetwork
    # # # Right now - disabled
    # # # Input image
    # im_in = Input(shape=(224, 224, 3))  # was 200, 200,4
    #
    # # processing with squeezenet
    # x1 = modelsqueeze(im_in)
    #
    # # Flattening
    # x1 = Flatten()(x1)  # 1,1,1000 -> 1000
    # # x1 = GlobalAveragePooling2D()(x1)
    # # x1 = Activation("relu")(x1)
    #
    # # Dense, dropout
    # x1 = Dense(512, activation="relu")(x1)
    # x1 = BatchNormalization()(x1)
    # x1 = Dropout(0.1)(x1)
    # x1 = Dense(512, activation="relu")(x1)
    # x1 = BatchNormalization()(x1)
    # x1 = Dropout(0.1)(x1)
    # out_x = Dense(512, activation="relu")(x1)
    #
    # # Seems like getting features of image
    # # feat_x = Dense(300)(x1)
    # # L2 normalization of result
    # # feat_x = Lambda(lambda x: K.l2_normalize(x, axis=1))(feat_x)
    #
    # # Finalizing model
    # model_top = Model(inputs=[im_in], outputs=out_x)
    #
    # model_top.summary()
    # # -----------------------------------------------

    # Really simple network
    # IDK what to do with squeezenet, it just doesn't want to train
    # I'll build simple CNN so that I'll test work of siamese network
    # because there is definitely something wrong
    input = Input((224, 224, 3))

    x = Convolution2D(64, 7, 2, "valid")(input)
    x = Convolution2D(64, 3, 1, "valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Convolution2D(128, 3, 2, "valid")(x)
    x = Convolution2D(128, 3, 2, "valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Convolution2D(256, 5, 2, "valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64)(x)

    output = x

    simplemodel = Model(input, output)

    simplemodel.summary()

    # Siamese network
    # Two images for siamese neural network

    im_in1 = Input((224, 224, 3))  # was 200, 200, 4
    im_in2 = Input((224, 224, 3))  # was 200, 200, 4

    # Processing them through squeezenet
    feat_x1 = modelsqueeze(im_in1)
    # feat_x1 = simplemodel(im_in1)
    # feat_x1 = Dense(512, activation="selu")(feat_x1)
    # feat_x1 = BatchNormalization()(feat_x1)

    feat_x2 = modelsqueeze(im_in2)
    # feat_x2 = simplemodel(im_in2)
    # feat_x2 = Dense(512, activation="selu")(feat_x2)
    # feat_x2 = BatchNormalization()(feat_x2)

    # Calculating euclidean_distance between two images
    merge_layer = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [feat_x1, feat_x2]
    )

    # normal_layer = BatchNormalization()(merge_layer)
    output_layer = Dense(1, activation="sigmoid")(merge_layer)
    # output_layer = Lambda(treshold_function, output_shape=treshold_shape)(output_layer)
    # 3rd, and final model.
    # Input - two images.
    # Output - distance between two images
    # output_layer = merge_layer

    model_final = Model(inputs=[im_in1, im_in2], outputs=output_layer)

    model_final.summary()

    # Optimizer
    lr_schedule = ExponentialDecay(0.01, decay_steps=1, decay_rate=0.99)
    adam = Adam(learning_rate=1e-5)
    adagrad = Adagrad(learning_rate=0.01)
    adamw = AdamW()
    rms = RMSprop()

    sgd = SGD(learning_rate=4e-2, momentum=0.9)

    # Compiling network.
    # Loss function is custom
    model_final.compile(
        optimizer=sgd,  # type: ignore
        metrics=[
            accuracy,
            keras.metrics.BinaryAccuracy(threshold=0.8),
            keras.metrics.F1Score(average="weighted", threshold=0.8),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            # keras.metrics.MeanSquaredError(),
        ],
        loss=contrastive_loss,
        # loss="binary_crossentropy",
    )

    # Output plot model to img
    plot_model(model_final, show_shapes=True, expand_nested=True, to_file="plot.png")

    return model_final


def loadModel(path="./src/faceid_network.keras"):
    # model 01getNewModel()
    model = load_model(
        path,
        compile=False,
        custom_objects={
            "euclidean_distance": euclidean_distance,
            "fire": fire,
            "eucl_dist_output_shape": eucl_dist_output_shape,
        },
    )
    # model = tf.saved_model.load(
    #     "./src/faceid_network",
    # )
    adam = Adam(learning_rate=0.00006)
    sgd = SGD(learning_rate=1e-6, momentum=0.9)
    model.compile(  # type: ignore
        optimizer=sgd,
        loss=contrastive_loss,
        metrics=[accuracy, "f1_score", "precision", "recall"],
    )

    return model


def get_offsets(x1, x2, y1, y2):
    left, right, up, down = 0, 0, 0, 0
    is_resize = False

    if x2 - x1 <= 224:
        left = int((224 - x2 + x1) / 2)
        x1 -= left
        if x1 < 0:
            x1 = 0
        x2 = x1 + 224
        if x2 > 640:  # TODO: add normal limits instead of hardcoded
            x1 -= x2 - 640
            x2 = 640
    else:
        is_resize = True

    if y2 - y1 <= 224:
        down = int((224 - y2 + y1) / 2)
        y1 -= down
        if y1 < 0:
            y1 = 0
        y2 = y1 + 224
        if y2 > 360:
            y1 -= y2 - 360
            y2 = 360
    else:
        is_resize = True

    return x1, x2, y1, y2, is_resize


def hide():
    # def create_correct_couple(folder, validation) -> np.array:
    #     if folder == 0:
    #         if validation:
    #             i = random.randint(0, 28)
    #         else:
    #             i = random.randint(29, 35)
    #
    #         img1 = np.load(f"src/data/{i}.npy")
    #         img1 = cv2.resize(img1, (640, 360))
    #         with open(f"src/data/{i}.pos") as file:
    #             x1 = int(file.readline())
    #             x2 = int(file.readline())
    #             y1 = int(file.readline())
    #             y2 = int(file.readline())
    #             is_resize = bool(file.readline())
    #
    #         img1 = img1[y1:y2, x1:x2]
    #         if is_resize:
    #             # top = int((224 - y2 + y1) / 2)
    #             # bottom = abs((y1-top + 224) - y2)
    #             # left = int((224 - x2 + x1) / 2)
    #             # right = abs((x1-left + 224) - x2)
    #             img1 = cv2.resize(img1, (224, 224))
    #             # img1 = cv2.copyMakeBorder(img1, top, bottom, left, right, cv2.BORDER_REFLECT)
    #
    #         if validation:
    #             i = random.randint(0, 28)
    #         else:
    #             i = random.randint(29, 35)
    #
    #         img2 = np.load(f"src/data/{i}.npy")
    #         img2 = cv2.resize(img2, (640, 360))
    #
    #         with open(f"src/data/{i}.pos") as file:
    #             x1 = int(file.readline())
    #             x2 = int(file.readline())
    #             y1 = int(file.readline())
    #             y2 = int(file.readline())
    #             is_resize = bool(file.readline())
    #
    #         img2 = img2[y1:y2, x1:x2]
    #         if is_resize:
    #             # top = int((224 - y2 + y1) / 2)
    #             # bottom = abs((y1-top + 224) - y2)
    #             # left = int((224 - x2 + x1) / 2)
    #             # right = abs((x1-left + 224) - x2)
    #             img2 = cv2.resize(img2, (224, 224))
    #             # img2 = cv2.copyMakeBorder(img2, top, bottom, left, right, cv2.BORDER_REFLECT)
    #
    #         img1 = data_augmentation(img1)
    #         img2 = data_augmentation(img2)
    #
    #         # i = random.randint(0, 2)
    #   # if i >= 1:
    #         #     img1 = cv2.flip(img1, i-1)
    #         # i = random.randint(0, 1)
    #         # if i >= 1:
    #         #     img2 = cv2.flip(img2, i-1)
    #
    #         return np.array([img1, img2])
    #     else:
    #         if validation:
    #             j = random.choice(list(faces_valid.keys())[:15])
    #             img1 = Image.open("./src/data/datasets/img_align_celeba/" + np.random.choice(faces_valid[j]))
    #             img1 = np.asarray(img1)
    #             # img1 = cv2.resize(img1, (224, 224))
    #             img1 = cv2.copyMakeBorder(img1, 3, 3, 23, 23, cv2.BORDER_REFLECT)
    #             img2 = Image.open("./src/data/datasets/img_align_celeba/" + np.random.choice(faces_valid[j]))
    #             img2 = np.asarray(img2)
    #             # img2 = cv2.resize(img2, (224, 224))
    #             img2 = cv2.copyMakeBorder(img2, 3, 3, 23, 23, cv2.BORDER_REFLECT)
    #         else:
    #             j = random.choice(list(faces_train.keys())[:15])
    #             img1 = Image.open("./src/data/datasets/img_align_celeba/" + np.random.choice(faces_train[j]))
    #             img1 = np.asarray(img1)
    #             # img1 = cv2.resize(img1, (224, 224))
    #             img1 = cv2.copyMakeBorder(img1, 3, 3, 23, 23, cv2.BORDER_REFLECT)
    #             img2 = Image.open("./src/data/datasets/img_align_celeba/" + np.random.choice(faces_train[j]))
    #             img2 = np.asarray(img2)
    #             # img2 = cv2.resize(img2, (224, 224))
    #             img2 = cv2.copyMakeBorder(img2, 3, 3, 23, 23, cv2.BORDER_REFLECT)
    #
    #         img1 = data_augmentation(img1)
    #         img2 = data_augmentation(img2)
    #         # i = random.randint(0, 2)
    #         # if i >= 1:
    #         #     img1 = cv2.flip(img1, i-1)
    #         # i = random.randint(0, 1)
    #         # if i >= 1:
    #         #     img2 = cv2.flip(img2, i-1)
    #         return np.array([img1, img2])
    #
    #
    # def create_incorrect_couple(folder, validation) -> np.array:
    #     if folder == 1:
    #
    #         if validation:
    #             i = random.randint(0, 28)
    #         else:
    #             i = random.randint(29, 35)
    #
    #         img1 = np.load(f"src/data/{i}.npy")
    #         img1 = cv2.resize(img1, (640, 360))
    #         with open(f"src/data/{i}.pos") as file:
    #             x1 = int(file.readline())
    #             x2 = int(file.readline())
    #             y1 = int(file.readline())
    #             y2 = int(file.readline())
    #             is_resize = bool(file.readline())
    #
    #         img1 = img1[y1:y2, x1:x2]
    #         if is_resize:
    #             # top = int((224 - y2 + y1) / 2)
    #             # bottom = abs((y1-top + 224) - y2)
    #             # left = int((224 - x2 + x1) / 2)
    #             # right = abs((x1-left + 224) - x2)
    #             img1 = cv2.resize(img1, (224, 224))
    #             # img1 = cv2.copyMakeBorder(img1, top, bottom, left, right, cv2.BORDER_REFLECT)
    #
    #         if validation:
    #             j = random.choice(list(faces_valid.keys())[:15])
    #             img2 = Image.open("./src/data/datasets/img_align_celeba/" +
    #                 random.choice(
    #                     faces_valid[j]
    #                 )
    #             )
    #         else:
    #             j = random.choice(list(faces_train.keys())[:15])
    #             img2 = Image.open("./src/data/datasets/img_align_celeba/" +
    #                 random.choice(
    #                     faces_train[j]
    #                 )
    #             )
    #         img2 = np.asarray(img2)
    #         # img2 = cv2.resize(img2, (224, 224))
    #         img2 = cv2.copyMakeBorder(img2, 3, 3, 23, 23, cv2.BORDER_REFLECT)
    #
    #         img1 = data_augmentation(img1)
    #         img2 = data_augmentation(img2)
    #         # i = random.randint(0, 2)
    #         # if i >= 1:
    #         #     img1 = cv2.flip(img1, i-1)
    #         # i = random.randint(0, 1)
    #         # if i >= 1:
    #         #     img2 = cv2.flip(img2, i-1)
    #         return np.array([img1, img2])
    #     else:
    #         if validation:
    #             j = random.choice(list(faces_valid.keys())[:15])
    #             file1 = random.choice(faces_valid[j])
    #             img1 = Image.open("./src/data/datasets/img_align_celeba/" + file1)
    #             img1 = np.asarray(img1)
    #             # img1 = cv2.resize(img1, (224, 224))
    #             img1 = cv2.copyMakeBorder(img1, 3, 3, 23, 23, cv2.BORDER_REFLECT)
    #
    #             i = random.choice(list(faces_valid.keys())[:15])
    #             if i == j:
    #                 i = random.choice(list(faces_valid.keys())[:15])
    #
    #             file2 = random.choice(faces_valid[i])
    #             img2 = Image.open("./src/data/datasets/img_align_celeba/" + file2)
    #             img2 = np.asarray(img2)
    #             # img2 = cv2.resize(img2, (224, 224))
    #             img2 = cv2.copyMakeBorder(img2, 3, 3, 23, 23, cv2.BORDER_REFLECT)
    #         else:
    #             j = random.choice(list(faces_train.keys())[:15])
    #             file1 = random.choice(faces_train[j])
    #             img1 = Image.open("./src/data/datasets/img_align_celeba/" + file1)
    #             img1 = np.asarray(img1)
    #             # img1 = cv2.resize(img1, (224, 224))
    #             img1 = cv2.copyMakeBorder(img1, 3, 3, 23, 23, cv2.BORDER_REFLECT)
    #
    #             i = random.choice(list(faces_train.keys())[:15])
    #             if i == j:
    #                 i = random.choice(list(faces_train.keys())[:15])
    #
    #             file2 = random.choice(faces_train[i])
    #             img2 = Image.open("./src/data/datasets/img_align_celeba/" + file2)
    #             img2 = np.asarray(img2)
    #             # img2 = cv2.resize(img2, (224, 224))
    #             img2 = cv2.copyMakeBorder(img2, 3, 3, 23, 23, cv2.BORDER_REFLECT)
    #
    #         img1 = data_augmentation(img1)
    #         img2 = data_augmentation(img2)
    #         # i = random.randint(0, 2)
    #         # if i >= 1:
    #         #     img1 = cv2.flip(img1, i-1)
    #         # i = random.randint(0, 1)
    #         # if i >= 1:
    #         #     img2 = cv2.flip(img2, i-1)
    #         return np.array([img1, img2])
    pass


def create_correct_couple(dataset, identities, only_user):
    if only_user:
        j = "myface"
    else:
        j = random.choice(identities)
    base_path = "./src/data/"
    if j != "myface":
        base_path += "datasets/img_align_celeba/"
    file1 = base_path + np.random.choice(dataset[j])
    img1 = Image.open(file1)
    img1 = np.asarray(img1)
    if j != "myface":
        img1 = cv2.copyMakeBorder(img1, 3, 3, 23, 23, cv2.BORDER_REFLECT)
    file2 = base_path + np.random.choice(dataset[j])
    img2 = Image.open(file2)
    img2 = np.asarray(img2)
    if j != "myface":
        img2 = cv2.copyMakeBorder(img2, 3, 3, 23, 23, cv2.BORDER_REFLECT)

    # To prevent overfitting
    # img1 = data_augmentation(img1)
    # img2 = data_augmentation(img2)
    if img1.shape != (224, 224, 3) or img2.shape != (224, 224, 3):
        # print(
        #     f"\nWARNING: img1 shape or img2 shape is broken. Validation: {validation}"
        # )
        print(f"Key used: {j}")
        print(f"file1: {file1}, out_shape: {img1.shape}")
        print(f"file2: {file2}, out_shape: {img2.shape}")
    return np.array([img1, img2])


def create_incorrect_couple(dataset, identities, only_user):
    if only_user:
        j = "myface"
    else:
        j = random.choice(identities)
    file1 = random.choice(dataset[j])
    base_path = "./src/data/"
    if j != "myface":
        base_path += "datasets/img_align_celeba/"

    file1 = base_path + file1
    img1 = Image.open(file1)
    img1 = np.asarray(img1)
    if j != "myface":
        img1 = cv2.copyMakeBorder(img1, 3, 3, 23, 23, cv2.BORDER_REFLECT)

    i = random.choice(identities)
    if i == j:
        i = random.choice(identities)

    file2 = random.choice(dataset[i])
    base_path = "./src/data/"
    if i != "myface":
        base_path += "datasets/img_align_celeba/"

    file2 = base_path + file2
    img2 = Image.open(file2)
    img2 = np.asarray(img2)
    if i != "myface":
        img2 = cv2.copyMakeBorder(img2, 3, 3, 23, 23, cv2.BORDER_REFLECT)

    # To prevent overfitting
    # img1 = data_augmentation(img1)
    # img2 = data_augmentation(img2)
    if img1.shape != (224, 224, 3) or img2.shape != (224, 224, 3):
        print(
            f"\nWARNING: img1 shape or img2 shape is broken. Validation: {validation}"
        )
        print(f"file1: {file1}, out_shape: {img1.shape}")
        print(f"file2: {file2}, out_shape: {img2.shape}")
    return np.array([img1, img2])


def generate_batch(batch_size, with_invalid=False, validation=False):
    while 1:
        X = []  # Data of that images
        Y = (
            []
        )  # Labels that mark each of my captured images as true (of false, if I'm pre-train model)

        valid = True
        for i in range(batch_size):
            if valid or with_invalid is False:
                val = create_correct_couple(faces_train)
                X.append(val)
                Y.append(np.array([0.0]))
                valid = False
            else:
                val = create_incorrect_couple(faces_valid)
                X.append(val)
                Y.append(np.array([1.0]))
                valid = True

        X = np.asarray(X)
        Y = np.asarray(Y)

        yield [X[:, 0], X[:, 1]], Y


def create_couples(dataset, person_key, number_of_samples=50):
    # j = random.choice(list(dataset.keys())[:50])
    j = person_key
    base_path = "./src/data/"
    if j != "myface":
        base_path += "datasets/img_align_celeba/"

    max_val = len(dataset[person_key])

    X = []
    Y = []

    for i in range(max_val):
        k = i
        while k < max_val:
            file1 = base_path + dataset[j][i]
            img1 = Image.open(file1)
            img1 = np.asarray(img1)
            if j != "myface":
                img1 = cv2.copyMakeBorder(img1, 3, 3, 23, 23, cv2.BORDER_REFLECT)

            file2 = base_path + dataset[j][k]
            img2 = Image.open(file2)
            img2 = np.asarray(img2)
            if j != "myface":
                img2 = cv2.copyMakeBorder(img2, 3, 3, 23, 23, cv2.BORDER_REFLECT)

            X += [[img1, img2]]
            Y += [[1.0]]

            l = random.choice(list(dataset.keys()))
            if l == i:
                l = random.choice(list(dataset.keys()))

            file2 = random.choice(dataset[l])
            base_path2 = "./src/data/"
            if l != "myface":
                base_path2 += "datasets/img_align_celeba/"

            file2 = base_path2 + file2
            img2 = Image.open(file2)
            img2 = np.asarray(img2)
            if i != "myface":
                img2 = cv2.copyMakeBorder(img2, 3, 3, 23, 23, cv2.BORDER_REFLECT)

            X += [[img1, img2]]
            Y += [[0.0]]

            k += 1

    return X, Y


full_train_data = []
full_train_data_labels = []
full_test_data = []
full_test_data_labels = []
full_valid_data = []
full_valid_data_labels = []


def generate_full_batch(max_number_of_faces=2, max_size=1000, only_user=False):
    global full_train_data, full_train_data_labels, full_test_data, full_test_data_labels, full_valid_data, full_valid_data_labels

    valid = True
    print(
        f"Generating {max_size} pairs of images and splitting them, this may take a while (and some RAM ^_^)..."
    )

    train_keys = list(faces_train.keys())
    # train_identities = [
    #     random.choice(list(faces_train.keys())) for _ in range(max_number_of_faces)
    # ]
    train_identities = random.sample(train_keys, len(train_keys))[:max_number_of_faces]

    print(
        f"train identities: {train_identities}, their sizes: {[len(faces_train[x]) for x in train_identities]}"
    )

    # test_identities = train_identities.copy()
    test_identities = [
        random.choice(list(faces_test.keys())) for _ in range(max_number_of_faces)
    ]
    print(
        f"test identities: {test_identities}, their sizes: {[len(faces_test[x]) for x in test_identities]}"
    )

    valid_keys = list(faces_valid.keys())
    # valid_identities = [
    #     random.choice(list(faces_valid.keys())) for _ in range(max_number_of_faces)
    # ]
    valid_identities = random.sample(valid_keys, len(valid_keys))[:max_number_of_faces]
    print(
        f"valid identities: {valid_identities}, their sizes: {[len(faces_valid[x]) for x in valid_identities]}"
    )

    for _ in range(int(max_size * 0.7)):
        valid = random.choice([True, False])
        if valid:
            val = create_correct_couple(faces_train, train_identities, only_user)
            full_train_data.append(val)
            full_train_data_labels.append(np.array([0.0]))
            # valid = False
        else:
            val = create_incorrect_couple(faces_train, train_identities, only_user)
            full_train_data.append(val)
            full_train_data_labels.append(np.array([1.0]))
            # valid = True
    print("Train set done")

    for _ in range(int(max_size * 0.2)):
        valid = random.choice([True, False])
        if valid:
            val = create_correct_couple(faces_test, test_identities, only_user)
            full_test_data.append(val)
            full_test_data_labels.append(np.array([0.0]))
            # valid = False
        else:
            val = create_incorrect_couple(faces_test, test_identities, only_user)
            full_test_data.append(val)
            full_test_data_labels.append(np.array([1.0]))
            # valid = True
    print("Test set done")

    for _ in range(int(max_size * 0.1)):
        valid = random.choice([True, False])
        if valid:
            val = create_correct_couple(faces_valid, valid_identities, only_user)
            full_valid_data.append(val)
            full_valid_data_labels.append(np.array([0.0]))
            # valid = False
        else:
            val = create_incorrect_couple(faces_valid, valid_identities, only_user)
            full_valid_data.append(val)
            full_valid_data_labels.append(np.array([1.0]))
            # valid = True
    print("Valid set done")

    # for key in list(faces_train.keys())[:max_number_of_faces]:
    #     X, Y = create_couples(faces_train, key)
    #     full_train_data = X
    #     full_train_data_labels = Y
    # print("Done train data")
    # for key in list(faces_test.keys())[:max_number_of_faces]:
    #     X, Y = create_couples(faces_test, key)
    #     full_test_data = X
    #     full_test_data_labels = Y
    # print("Done test data")
    # for key in list(faces_valid.keys())[:max_number_of_faces]:
    #     X, Y = create_couples(faces_valid, key)
    #     full_valid_data = X
    #     full_valid_data_labels = Y
    # print("Done valid data")

    # For further batch support
    # Without it you'll get an error
    full_train_data = np.array(full_train_data)
    full_train_data_labels = np.array(full_train_data_labels)
    full_test_data = np.array(full_test_data)
    full_test_data_labels = np.array(full_test_data_labels)
    full_valid_data = np.array(full_valid_data)
    full_valid_data_labels = np.array(full_valid_data_labels)

    print("Generating done. Resuming training.")


def get_pairs(num_classes_train=64, num_classes_test_percent=10):
    global full_train_data, full_train_data_labels, full_test_data, full_test_data_labels

    train_keys = list(faces_train.keys())
    train_identities = random.sample(train_keys, len(train_keys))[:num_classes_train]

    print(
        f"train identities: {train_identities}, their sizes: {[len(faces_train[x]) for x in train_identities]}"
    )

    print(f"train and test keys will be: {train_identities}")
    # print(f"test keys will be: {train_identities[num_classes_train-num_classes_test:]}")

    # Loading all required images in memory
    faces = []
    faces_labels = []
    base_path = "./src/data/datasets/img_align_celeba/"

    for identity in train_identities:
        for face in faces_train[identity][:5]:
            img = Image.open(base_path + face)
            img = np.asarray(img)
            img = cv2.resize(img, (224, 224))
            # img = cv2.copyMakeBorder(img, 3, 3, 23, 23, cv2.BORDER_REFLECT)
            faces.append(img)
            faces_labels.append(identity)

    # Preparing pairs
    tuples = [(x, y) for x, y in zip(faces, faces_labels)]

    for t in itertools.product(tuples, tuples):
        img1, label1 = t[0]
        img2, label2 = t[1]

        new_label = float(label1 != label2)

        full_train_data.append([img1, img2])
        full_train_data_labels.append([new_label])

    full_train_data, full_train_data_labels = shuffle(
        full_train_data, full_train_data_labels, random_state=0
    )

    full_test_data = np.array(
        full_train_data[
            int(len(full_train_data) * (1 - num_classes_test_percent / 100)) :
        ]
    )
    full_test_data_labels = np.array(
        full_train_data_labels[
            int(len(full_train_data_labels) * (1 - num_classes_test_percent / 100)) :
        ]
    )

    full_train_data = np.array(
        full_train_data[
            : int(len(full_train_data) * (1 - num_classes_test_percent / 100))
        ]
    )
    full_train_data_labels = np.array(
        full_train_data_labels[
            : int(len(full_train_data_labels) * (1 - num_classes_test_percent / 100))
        ]
    )

    print(
        full_train_data.shape,
        full_train_data_labels.shape,
        full_test_data.shape,
        full_test_data_labels.shape,
    )

    # # Now for test data
    # faces = []
    # faces_labels = []

    # for identity in train_identities[num_classes_train - num_classes_test :]:
    #     for face in faces_train[identity][:5]:
    #         img = Image.open(base_path + face)
    #         img = np.asarray(img)
    #         img = cv2.copyMakeBorder(img, 3, 3, 23, 23, cv2.BORDER_REFLECT)
    #         faces.append(img)
    #         faces_labels.append(identity)

    # # Preparing pairs
    # tuples = [(x, y) for x, y in zip(faces, faces_labels)]

    # for t in itertools.product(tuples, tuples):
    #     img1, label1 = t[0]
    #     img2, label2 = t[1]

    #     new_label = int(label1 == label2)

    #     full_test_data.append([img1, img2])
    #     full_test_data_labels.append([new_label])

    # full_test_data = np.array(full_test_data)
    # full_test_data_labels = np.array(full_test_data_labels)


def visualize(
    pairs,
    labels,
    filename="test.png",
    to_show=6,
    num_col=3,
    predictions=None,
    test=False,
):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.

    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).

    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):
        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(np.concatenate([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("true: {} | pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 5.9, 5.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(filename)


def train(model, event):
    refresh_user_pictures()
    # get_pairs(20, 5)
    generate_full_batch(4000, 100, True)
    # with open("dataset.dst", 'wb') as file:
    #     pickle.dump(file, [full_train_data, full_train_data_labels, full_test_data, full_test_data_labels, full_valid_data, full_valid_data_labels])

    # with open("dataset.dst", "rb") as file:
    #     full_train_data, full_train_data_labels, full_test_data, full_test_data_labels, full_valid_data, full_valid_data_labels = pickle.load(file)

    # print(f"Shape of full_train_data: {full_train_data.shape}, full_test_data: {full_test_data.shape}, full_valid_data: {full_valid_data.shape}")
    visualize(
        full_train_data[:-1],
        full_train_data_labels[:-1],
        to_show=4,
        num_col=2,
        filename="train.png",
    )
    visualize(
        full_test_data[:-1],
        full_test_data_labels[:-1],
        to_show=4,
        num_col=2,
        filename="test.png",
    )
    # visualize(
    #     full_valid_data[:-1],
    #     full_valid_data_labels[:-1],
    #     to_show=4,
    #     num_col=2,
    #     filename="valid.png",
    # )

    # data = generate_batch(40, True, False)
    # val_data = generate_batch(10, True, True)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_f1_score",
        patience=5,
        verbose=1,
        start_from_epoch=3,
        min_delta=1e-3,
    )
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath="./src/faceid_network.keras", save_best_only=True
    )

    new_model = model.fit(
        # data,
        [full_train_data[:, 0], full_train_data[:, 1]],
        full_train_data_labels,
        # steps_per_epoch=100,
        batch_size=10,
        epochs=50,
        validation_data=(
            [full_test_data[:, 0], full_test_data[:, 1]],
            full_test_data_labels,
        ),
        # validation_steps=1,
        callbacks=[
            checkpointer,
            early_stop,
        ],
    )

    # fig = plt.figure()
    fig, axarr = plt.subplots(
        10, sharex=True, figsize=(15.0, 30.0)
    )  # Figsize in inches

    axarr[0].plot(new_model.history["accuracy"])
    axarr[1].plot(new_model.history["loss"])
    axarr[2].plot(new_model.history["f1_score"])
    axarr[3].plot(new_model.history["precision"])
    axarr[4].plot(new_model.history["recall"])
    axarr[5].plot(new_model.history["val_accuracy"])
    axarr[6].plot(new_model.history["val_loss"])
    axarr[7].plot(new_model.history["val_f1_score"])
    axarr[8].plot(new_model.history["val_precision"])
    axarr[9].plot(new_model.history["val_recall"])
    plt.title("Model Metrics")
    plt.xlabel("Epoch")
    axarr[0].set_ylabel("Accuracy")
    axarr[1].set_ylabel("Loss")
    axarr[2].set_ylabel("F1 Score")
    axarr[3].set_ylabel("Presicion")
    axarr[4].set_ylabel("Recall")
    axarr[5].set_ylabel("Val Accuracy")
    axarr[6].set_ylabel("Val Loss")
    axarr[7].set_ylabel("Val F1 Score")
    axarr[8].set_ylabel("Val Precision")
    axarr[9].set_ylabel("Val Recall")
    # plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'], loc='upper left')

    plt.tight_layout()

    plt.savefig("history.png")
    plt.savefig("history1.png", dpi=300)
    plt.savefig("history2.png", dpi=200)
    plt.savefig("history3.png", dpi=100)
    plt.savefig("history4.png", dpi=50)
    plt.savefig("history5.png", dpi=25)

    plt.clf()

    model.save("./src/faceid_network.keras")
    event.set()


def modelDetectFace(model, frame2):
    i = random.randint(0, 35)
    frame1 = Image.open(f"src/data/{i}.jpg")
    frame1 = np.array(frame1)
    # img1.thumbnail((640, 360))
    # frame1 = cv2.resize(frame1, (640, 360))
    # img1 = np.asarray(img1)
    # _, _, _, x_coords, y_coords = detectFace(img1)
    #
    # x1, x2 = x_coords
    # y1, y2 = y_coords
    #
    # x1, x2, y1, y2 = get_offsets(x1, x2, y1, y2)
    #
    # with open(f"./src/data/{i}.pos") as file:
    #     x1 = int(file.readline())
    #     x2 = int(file.readline())
    #     y1 = int(file.readline())
    #     y2 = int(file.readline())
    #
    # frame1 = frame1[y1:y2, x1:x2]

    # frame2 = cv2.resize(frame2, (640, 360))
    # _, _, detected, x_coords, y_coords = detectFace(frame2)

    # if detected:
    #     x1, x2 = x_coords
    #     y1, y2 = y_coords

    #     x1, x2, y1, y2, is_resize = get_offsets(x1, x2, y1, y2)
    #     if (
    #         x1 >= 640
    #         or x1 <= 0
    #         or x2 >= 640
    #         or x2 <= 0
    #         or y1 >= 360
    #         or y1 <= 0
    #         or y2 >= 360
    #         or y2 <= 0
    #     ):
    #         return False, [[2.0]]

    #     frame2 = frame2[y1:y2, x1:x2]
    #     if is_resize:
    #         frame2 = cv2.resize(frame2, (224, 224))
    # else:
    #     # Return false, because face is not detected on a frame
    #     return False, [[2.0]]

    frame2 = cv2.resize(frame2, (224, 224))

    res = model.predict([np.array([frame1]), np.array([frame2])])
    print(res)
    if res < 1.0:
        return True, res
    return False, res


if __name__ == "__main__":
    model = loadModel()

    # img_input = Input(shape=(224, 224, 3))  # was 200, 200, 4

    # x = Convolution2D(64, (7, 7), strides=2, padding="same")(img_input)
    # x = Activation("relu")(x)
    # x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # x = BatchNormalization()(x)

    # x = fire(x, squeeze=16, expand=64)

    # x = fire(x, squeeze=16, expand=64)

    # x = fire(x, squeeze=32, expand=128)

    # x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # x = Activation("relu")(x)

    # x = fire(x, squeeze=32, expand=128)

    # x = fire(x, squeeze=48, expand=192)

    # x = fire(x, squeeze=48, expand=192)

    # x = fire(x, squeeze=64, expand=256)

    # x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # x = Activation("relu")(x)

    # x = fire(x, squeeze=64, expand=256)

    # x = Dropout(0.5)(x)  # Only if learning the SqueezeNet itself

    # x = Convolution2D(1000, (1, 1), strides=1)(x)
    # # x = BatchNormalization()(x)

    # x = GlobalAveragePooling2D()(
    #     x
    # )  # I believe this layer is obsolete for a siamese network
    # # # Because there will be calculated eu

    # x = Dense(64, activation="relu")(x)

    # out = x

    # modelsqueeze = Model(img_input, out)

    # import visualkeras

    # visualkeras.layered_view(modelsqueeze, to_file="modelview_faceid1.png")

    get_pairs(20, 5)
    results = model.evaluate(
        [full_train_data[:, 0], full_train_data[:, 1]],
        full_train_data_labels,
        batch_size=1,
    )
    print(results)
    # y_pred = model.predict([full_test_data[:, 0], full_test_data[:, 1]])
    # y_true = full_test_data_labels

    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # import matplotlib.pyplot as plt

    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # disp.plot(cmap=plt.cm.Blues)
    # plt.show()

# matrix = tf.math.confusion_matrix(y_true, y_pred)
