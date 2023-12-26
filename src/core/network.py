# Code is blatantly stolen from github.
# I'll learn it a little bit later
# and probably rewrite it
# because I want to have full understanding of what is happening here
# and most importantly, how and why.

from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Flatten,
    Dropout,
    Lambda,
    ELU,
    concatenate,
    GlobalAveragePooling2D,
    Input,
    BatchNormalization,
    SeparableConv2D,
    Subtract,
    concatenate,
)
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Conv2D as Convolution2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD  # , RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy as SCC

# from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model, split_dataset
import tensorflow as tf

import numpy as np
import glob
import random
import cv2
from matplotlib import pyplot as plt

from PIL import Image

from core.headPosition import detectFace

gpus = tf.config.list_physical_devices('GPU')
if gpus: 
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3692)]
    )

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

prepared_images_all = glob.glob(f"src/data/datasets/prepared/*.png")
prep_left, prep_right = split_dataset(np.array(prepared_images_all), left_size=0.7)
prep_left = [val.numpy() for val in prep_left]
prep_right = [val.numpy() for val in prep_right]


prepared_images = {}
for i in range(1, 11):
    left, right = split_dataset(np.array(glob.glob(f"src/data/datasets/prepared/{i:03d}_*.png")), left_size = 0.7)
    prepared_images[i] = {}
    prepared_images[i][0] = [val.numpy() for val in left]
    prepared_images[i][1] = [val.numpy() for val in right]


def euclidean_distance(inputs):
    assert len(inputs) == 2, "Euclidean distance needs 2 inputs, %d given" % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred):
    margin = 1.0
    return K.mean(
        (1.0 - y_true) * K.square(y_pred)
        + y_true * K.square(K.maximum(margin - y_pred, 0.0))
    )


# return K.mean( K.square(y_pred) )
emb_size=10
alpha=1

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size],y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)


# --------------
def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1, 1), padding="valid")(x)
    x = Activation("relu")(x)

    left = Convolution2D(expand, (1, 1), padding="valid")(x)
    left = Activation("relu")(left)

    right = Convolution2D(expand, (3, 3), padding="same")(x)
    right = Activation("relu")(right)

    x = concatenate([left, right], axis=3)
    return x


def getNewModel2():
    
    # SqueezeNet architecture

    img_input = Input(shape=(224, 224, 3))  # was 200, 200, 4

    x = Convolution2D(96, (7, 7), strides=2, padding="same")(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = fire(x, squeeze=16, expand=64)

    x = fire(x, squeeze=16, expand=64)

    x = fire(x, squeeze=32, expand=128)

    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = fire(x, squeeze=32, expand=128)

    x = fire(x, squeeze=48, expand=192)

    x = fire(x, squeeze=48, expand=192)

    x = fire(x, squeeze=64, expand=256)

    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = fire(x, squeeze=64, expand=256)

    x = Dropout(0.001)(x)

    x = Convolution2D(1000, (1, 1), strides=1)(x)
    # x = Activation("relu")(x)
    out = AveragePooling2D(pool_size=(13, 13), strides=(1, 1))(x)
    out = Activation("relu")(out)

    modelsqueeze = Model(img_input, out)

    modelsqueeze.summary()

    # SqueezeNet end

    # Second neural model - Siamese network
    # Input image
    im_in = Input(shape=(224, 224, 3))  # was 200, 200,4

    # processing with squeezenet
    x1 = modelsqueeze(im_in)

    # Flattening
    x1 = Flatten()(x1)  # 1,1,1000 -> 1000

    # Dense, dropout
    x1 = Dense(384)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("softmax")(x1)
    x1 = Dropout(0.1)(x1)
    # Seems like getting features of image
    feat_x = Dense(30, activation="linear")(x1)
    # L2 normalization of result
    feat_x = Lambda(lambda x: K.l2_normalize(x, axis=1))(feat_x)

    # Finalizing model
    model_top = Model(inputs=[im_in], outputs=feat_x)

    model_top.summary()

    # Two images for siamese neural networki

    im_in1 = Input(shape=(224, 224, 3))  # was 200, 200,4
    im_in2 = Input(shape=(224, 224, 3))  # was 200, 200,4

    feat_x1 = model_top(im_in1)
    # feat_x1 = Activation("relu")(feat_x1)
    feat_x2 = model_top(im_in2)
    # feat_x2 = Activation("relu")(feat_x2)

    # Calculating euclidean_distance between two images
    lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])

    # 3rd, and final model.
    # Input - two images.
    # Output - distance between two images

    model_final = Model(inputs=[im_in1, im_in2], outputs=lambda_merge)

    model_final.summary()

    # Optimizer
    lr_schedule = ExponentialDecay(0.001, decay_steps=100000, decay_rate=0.96)
    adam = Adam(learning_rate=lr_schedule)
    # adam = Adam(learning_rate=0.001)

    sgd = SGD(learning_rate=0.001, momentum=0.9)

    # Compiling network.
    # Loss function is custom
    model_final.compile(optimizer=adam, metrics=['accuracy'], loss=contrastive_loss)

    # Output plot model to img
    # plot_model(model_final, show_shapes=True, expand_nested=True, to_file="plot.png")

    return model_final


def getNewModel():
    img_input = Input(shape=(224, 224, 3))  # was 200, 200, 4

    x = Convolution2D(64, (5, 5), strides=(2, 2), padding="valid")(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire(x, squeeze=16, expand=16)

    x = fire(x, squeeze=16, expand=16)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire(x, squeeze=32, expand=32)

    x = fire(x, squeeze=32, expand=32)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire(x, squeeze=48, expand=48)

    x = fire(x, squeeze=48, expand=48)

    x = fire(x, squeeze=64, expand=64)

    x = fire(x, squeeze=64, expand=64)

    x = Dropout(0.2)(x)

    x = Convolution2D(512, (1, 1), padding="same")(x)
    out = Activation("relu")(x)

    modelsqueeze = Model(img_input, out)

    modelsqueeze.summary()

    im_in = Input(shape=(224, 224, 3))  # was 200, 200,4
    # wrong = Input(shape=(130,200,3))

    x1 = modelsqueeze(im_in)
    # x = Convolution2D(64, (5, 5), padding='valid', strides =(2,2))(x)

    # x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x1)

    """
    x1 = Convolution2D(256, (3,3), padding='valid', activation="relu")(x1)
    x1 = Dropout(0.4)(x1)

    x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x1)

    x1 = Convolution2D(256, (3,3), padding='valid', activation="relu")(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.4)(x1)

    x1 = Convolution2D(64, (1,1), padding='same', activation="relu")(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.4)(x1)
    """

    x1 = Flatten()(x1)

    x1 = Dense(512, activation="relu")(x1)
    x1 = Dropout(0.2)(x1)
    # x1 = BatchNormalization()(x1)
    feat_x = Dense(128, activation="linear")(x1)
    feat_x = Lambda(lambda x: K.l2_normalize(x, axis=1))(feat_x)

    model_top = Model(inputs=[im_in], outputs=feat_x)

    model_top.summary()

    im_in1 = Input(shape=(224, 224, 3))  # was 200, 200,4
    im_in2 = Input(shape=(224, 224, 3))  # was 200, 200,4

    feat_x1 = model_top(im_in1)
    feat_x2 = model_top(im_in2)

    lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])

    model_final = Model(inputs=[im_in1, im_in2], outputs=lambda_merge)

    model_final.summary()

    adam = Adam(learning_rate=0.001)

    sgd = SGD(learning_rate=0.001, momentum=0.9)

    model_final.compile(optimizer=adam, loss=contrastive_loss)

    # plot_model(model_final, show_shapes=True, expand_nested=True, to_file="plot.png")

    return model_final


def loadModel():
    # model 01getNewModel()
    model = load_model("src/faceid_network/", compile=False)
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss=contrastive_loss)

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


def create_correct_couple(folder, validation) -> np.array:
    if folder == 0:
        i = random.randint(0, 35)
        img1 = np.load(f"src/data/{i}.npy")
        # img1.thumbnail((640, 360))
        img1 = cv2.resize(img1, (640, 360))
        # img1.save("thumbnail.jpg", "JPEG")
        # img1 = np.asarray(img1)
        # _, _, detected, x_coords, y_coords = detectFace(img1)
        # while not detected:
        #     print(f'{i}: {detected}')
        #     _, _, detected, x_coords, y_coords = detectFace(img1)
        #
        #
        # x1, x2 = x_coords
        # y1, y2 = y_coords
        #
        # x1, x2, y1, y2 = get_offsets(x1, x2, y1, y2)
        #
        # print(f'{i}, {detected}, {x_coords}, {y_coords}, {img1.shape}')
        with open(f"src/data/{i}.pos") as file:
            x1 = int(file.readline())
            x2 = int(file.readline())
            y1 = int(file.readline())
            y2 = int(file.readline())
            is_resize = bool(file.readline())

        img1 = img1[y1:y2, x1:x2]
        if is_resize:
            img1 = cv2.resize(img1, (224, 224))

        # print(f'{i} ', end='')

        i = random.randint(0, 35)
        img2 = np.load(f"src/data/{i}.npy")
        # img2.thumbnail((640, 480))
        img2 = cv2.resize(img2, (640, 360))
        # img2 = np.asarray(img2)
        # _, _, detected, x_coords, y_coords = detectFace(img2)
        #
        # while not detected:
        #     print(f'{i}: {detected}')
        #     _, _, detected, x_coords, y_coords = detectFace(img2)
        #
        # x1, x2 = x_coords
        # y1, y2 = y_coords
        #
        # x1, x2, y1, y2 = get_offsets(x1, x2, y1, y2)

        # print(f'{i}, {detected}, {x_coords}, {y_coords}, {img2.shape}')
        with open(f"src/data/{i}.pos") as file:
            x1 = int(file.readline())
            x2 = int(file.readline())
            y1 = int(file.readline())
            y2 = int(file.readline())
            is_resize = bool(file.readline())

        img2 = img2[y1:y2, x1:x2]
        if is_resize:
            img2 = cv2.resize(img2, (224, 224))

        # print(f'{i}: {img1.shape}, {img2.shape}')

        return np.array([img1, img2])
    else:
        if validation:
            i = 1
        else:
            i = 0
        j = random.randint(1, 10)
        img1 = Image.open(np.random.choice(prepared_images[j][i]))
        img1 = np.asarray(img1)
        img2 = Image.open(np.random.choice(prepared_images[j][i]))
        img2 = np.asarray(img2)

        return np.array([img1, img2])


def create_incorrect_couple(folder, validation) -> np.array:
    # folder = np.random.randint(2)
    if folder == 1 or folder == 3:
        i = random.randint(0, 35)
        img1 = np.load(f"src/data/{i}.npy")
        # img1.thumbnail((640, 360))
        img1 = cv2.resize(img1, (640, 360))
        with open(f"src/data/{i}.pos") as file:
            x1 = int(file.readline())
            x2 = int(file.readline())
            y1 = int(file.readline())
            y2 = int(file.readline())
            is_resize = bool(file.readline())

        img1 = img1[y1:y2, x1:x2]
        if is_resize:
            img1 = cv2.resize(img1, (224, 224))

        # i = random.randint(0, 25)
        # j = random.randint(0, 12)
        # img1 = Image.open(np.random.choice(glob.glob(f"data/datasets/RGBD_Face_dataset_training/*_image_corrected.png")))
        # img1.thumbnail((640, 360))
        # img1 = np.asarray(img1)
        # _, _, detected, x_coords, y_coords = detectFace(img1)
        #
        # while not detected:
        #     print(f'{i},{j} - {detected}')
        #     i = random.randint(0, 25)
        #     j = random.randint(0, 12)
        #     img1 = Image.open(f"data/datasets/RGBD_Face_dataset_training/{i:03d}_{j:02d}_image.png")
        #     img1.thumbnail((640, 360))
        #     img1 = np.asarray(img1)
        #     _, _, detected, x_coords, y_coords = detectFace(img1)
        #
        # x1, x2 = x_coords
        # y1, y2 = y_coords
        #
        # x1, x2, y1, y2 = get_offsets(x1, x2, y1, y2)

        # Approximated coordinates, because mediapipe cannot detect faces propertly...
        # x1, x2, y1, y2 = 200, 400, 80, 280

        # img1 = img1[y1:y2, x1:x2]

        # i = random.randint(0, 25)
        # j = random.randint(0, 12)
        if validation:
            img2 = Image.open(
                np.random.choice(
                    prep_right
                )
            )
        else:
            img2 = Image.open(
                np.random.choice(
                    prep_left
                )
            )
        # j = random.randint(1, 10)
        # img2 = Image.open(
        #     np.random.choice(
        #         
        #     )
        # )
        # img2.thumbnail((640, 360))
        img2 = np.asarray(img2)
        # _, _, detected, x_coords, y_coords = detectFace(img2)
        #
        # while not detected:
        #     print(f'{i},{j} - {detected}')
        #     i = random.randint(0, 25)
        #     j = random.randint(0, 12)
        #     img2 = Image.open(f"data/datasets/RGBD_Face_dataset_training/{i:03d}_{j:02d}_image.png")
        #     img2.thumbnail((640, 360))
        #     img2 = np.asarray(img2)
        #     _, _, detected, x_coords, y_coords = detectFace(img2)
        #
        # x1, x2 = x_coords
        # y1, y2 = y_coords
        #
        # x1, x2, y1, y2 = get_offsets(x1, x2, y1, y2)

        # img2 = img2[y1:y2, x1:x2]

        # print(f'{img1.shape}, {img2.shape}')

        return np.array([img1, img2])
    else:
        # i = random.randint(1, 10)
        if validation:
            i = 1
        else:
            i = 0
        k = random.randint(1, 10)
        file1 = np.random.choice(prepared_images[k][i])
        img1 = Image.open(file1)
        img1 = np.asarray(img1)
        l = random.randint(1, 10)
        while l != k:
            l = random.randint(1, 10)
        #     if validation:
        #         j = random.randint(8, 10)
        #     else:
        #         j = random.randint(1, 7)
        file2 = np.random.choice(prepared_images[l][i])
        # while file2 == file1:
        #     file2 = np.random.choice(prepared_images[j])
        img2 = Image.open(file2)
        img2 = np.asarray(img2)

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
                val = create_correct_couple(i%4, validation)
                X.append(val)
                Y.append(np.array([0.0]))
                valid = False
                # valid = False if np.random.random() <= 0.5 else True
            else:
                val = create_incorrect_couple(i%4, validation)
                X.append(val)
                Y.append(np.array([1.0]))
                valid = True
                # valid = False if np.random.random() <= 0.5 else True

        X = np.asarray(X)
        Y = np.asarray(Y)

        yield [X[:, 0], X[:, 1]], Y


def train(model, event):
    data = generate_batch(40, True, False)
    val_data = generate_batch(16, True, True)
    new_model = model.fit(
        data,
        steps_per_epoch=10,
        epochs=50,
        validation_data=val_data,
        validation_steps=10,
    )

    fig = plt.figure()

    plt.plot(new_model.history['loss'])
    plt.plot(new_model.history['val_loss'])
    plt.plot(new_model.history['accuracy'])
    plt.plot(new_model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'], loc='upper left')

    fig.set_size_inches(30., 18.)

    plt.savefig('history.png', dpi=300)
    plt.savefig('history2.png', dpi=200)
    plt.savefig('history3.png', dpi=100)
    plt.savefig('history4.png', dpi=50)
    plt.savefig('history5.png', dpi=25)

    model.save("./src/faceid_network/")
    event.set()


def modelDetectFace(model, frame2):
    i = random.randint(0, 35)
    frame1 = np.load(f"src/data/{i}.npy")
    # img1.thumbnail((640, 360))
    frame1 = cv2.resize(frame1, (640, 360))
    # img1 = np.asarray(img1)
    # _, _, _, x_coords, y_coords = detectFace(img1)
    #
    # x1, x2 = x_coords
    # y1, y2 = y_coords
    #
    # x1, x2, y1, y2 = get_offsets(x1, x2, y1, y2)
    #
    with open(f"./src/data/{i}.pos") as file:
        x1 = int(file.readline())
        x2 = int(file.readline())
        y1 = int(file.readline())
        y2 = int(file.readline())

    frame1 = frame1[y1:y2, x1:x2]

    frame2 = cv2.resize(frame2, (640, 360))
    _, _, detected, x_coords, y_coords = detectFace(frame2)

    if detected:
        x1, x2 = x_coords
        y1, y2 = y_coords

        x1, x2, y1, y2, is_resize = get_offsets(x1, x2, y1, y2)
        if (
            x1 >= 640
            or x1 <= 0
            or x2 >= 640
            or x2 <= 0
            or y1 >= 360
            or y1 <= 0
            or y2 >= 360
            or y2 <= 0
        ):
            return False, [[2.0]]

        frame2 = frame2[y1:y2, x1:x2]
        if is_resize:
            frame2 = cv2.resize(frame2, (224, 224))
    else:
        # Return false, because face is not detected on a frame
        return False, [[2.0]]

    res = model.predict([np.array([frame1]), np.array([frame2])])
    print(res)
    if res <= 0.15:
        return True, res
    return False, res
