# Code is blatantly stolen from github.
# I'll learn it a little bit later
# and probably rewrite it
# because I want to have full understanding of what is happening here
# and most importantly, how and why.

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))
        

def contrastive_loss(y_true,y_pred):
    margin=1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))
   # return K.mean( K.square(y_pred) )

def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1,1), padding='valid')(x)
    x = Activation('relu')(x)
    
    left = Convolution2D(expand, (1,1), padding='valid')(x)
    left = Activation('relu')(left)
    
    right = Convolution2D(expand, (3,3), padding='same')(x)
    right = Activation('relu')(right)
    
    x = concatenate([left, right], axis=3)
    return x

img_input=Input(shape=(200,200,4))

x = Convolution2D(64, (5, 5), strides=(2, 2), padding='valid')(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
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

x = Convolution2D(512, (1, 1), padding='same')(x)
out = Activation('relu')(x)


modelsqueeze= Model(img_input, out)

modelsqueeze.summary()

im_in = Input(shape=(200,200,4))
#wrong = Input(shape=(200,200,3))

x1 = modelsqueeze(im_in)
#x = Convolution2D(64, (5, 5), padding='valid', strides =(2,2))(x)

#x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x1)

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
#x1 = BatchNormalization()(x1)
feat_x = Dense(128, activation="linear")(x1)
feat_x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_x)


model_top = Model(inputs = [im_in], outputs = feat_x)

model_top.summary()

im_in1 = Input(shape=(200,200,4))
im_in2 = Input(shape=(200,200,4))

feat_x1 = model_top(im_in1)
feat_x2 = model_top(im_in2)


lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])


model_final = Model(inputs = [im_in1, im_in2], outputs = lambda_merge)

model_final.summary()

adam = Adam(lr=0.001)

sgd = SGD(lr=0.001, momentum=0.9)

model_final.compile(optimizer=adam, loss=contrastive_loss)

