import tensorflow as tf
import keras.backend as K
import typing
from keras.models import Input
from keras.engine.training import Model
from keras.layers import concatenate
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, LeakyReLU, Conv2DTranspose
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np


###https://github.com/longuyen97/UnetPlusPlus/blob/master/model/unetpp.py#L215

IMAGE_SIZE = (256,256,1)

batch_size = 2
epochs = 25
number_of_filters = 2
num_classes = 1
weights_list = {1: 1.0, 2: 50.0, 3: 70.0}

def weighted_loss(original_loss_function: typing.Callable, weights_list: dict) -> typing.Callable:
    def loss_function(true, pred):
        class_selectors = tf.cast(K.argmax(true, axis=-1), tf.int32)
        class_selectors = [K.equal(i, class_selectors) for i in range(len(weights_list))]
        class_selectors = [K.cast(x, K.floatx()) for x in class_selectors]
        weights = [sel * w for sel, w in zip(class_selectors, weights_list)]
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]
        loss = original_loss_function(true, pred)
        loss = loss * weight_multiplier
        return loss
    return loss_function


@tf.function
def cce_iou_dice(y_true, y_pred, smooth=1, cat_weight=1, iou_weight=1, dice_weight=1):
    return cat_weight * K.categorical_crossentropy(y_true, y_pred) \
           + iou_weight * log_iou(y_true, y_pred, smooth) \
           + dice_weight * log_dice(y_true, y_pred, smooth)


@tf.function
def cce_iou(y_true, y_pred, smooth=1, cat_weight=1, iou_weight=1):
    return cat_weight * K.categorical_crossentropy(y_true, y_pred) + iou_weight * log_iou(y_true, y_pred, smooth)


@tf.function
def cce_dice(y_true, y_pred, smooth=1, cat_weight=1, dice_weight=1):
    return cat_weight * K.categorical_crossentropy(y_true, y_pred) + dice_weight * log_dice(y_true, y_pred, smooth)


@tf.function
def bce_iou(y_true, y_pred, smooth=1, cat_weight=1, iou_weight=1):
    return cat_weight * K.binary_crossentropy(y_true, y_pred) + iou_weight * log_iou(y_true, y_pred, smooth)


@tf.function
def bce_dice(y_true, y_pred, smooth=1, cat_weight=1, dice_weight=1):
    return cat_weight * K.binary_crossentropy(y_true, y_pred) + dice_weight * log_dice(y_true, y_pred, smooth)


@tf.function
def log_iou(y_true, y_pred, smooth=1):
    return - K.log(iou(y_true, y_pred, smooth))


@tf.function
def log_dice(y_true, y_pred, smooth=1):
    return -K.log(dice(y_true, y_pred, smooth))


@tf.function
def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


@tf.function
def dice(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def conv2d(filters: int):
    return Conv2D(filters=filters,
                  kernel_size=(3, 3),
                  padding='same',
                  kernel_regularizer=l2(0.),
                  bias_regularizer=l2(0.))


def conv2dtranspose(filters: int):
    return Conv2DTranspose(filters=filters,
                           kernel_size=(2, 2),
                           strides=(2, 2),
                           padding='same')


def model(weights_input=None):
    inputs = Input(IMAGE_SIZE)
    x00 = conv2d(filters=int(16 * number_of_filters))(inputs)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    x00 = Dropout(0.2)(x00)
    x00 = conv2d(filters=int(16 * number_of_filters))(x00)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    x00 = Dropout(0.2)(x00)
    p0 = MaxPooling2D(pool_size=(2, 2))(x00)

    x10 = conv2d(filters=int(32 * number_of_filters))(p0)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    x10 = Dropout(0.2)(x10)
    x10 = conv2d(filters=int(32 * number_of_filters))(x10)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    x10 = Dropout(0.2)(x10)
    p1 = MaxPooling2D(pool_size=(2, 2))(x10)

    x01 = conv2dtranspose(int(16 * number_of_filters))(x10)
    x01 = concatenate([x00, x01])
    x01 = conv2d(filters=int(16 * number_of_filters))(x01)
    x01 = BatchNormalization()(x01)
    x01 = LeakyReLU(0.01)(x01)
    x01 = conv2d(filters=int(16 * number_of_filters))(x01)
    x01 = BatchNormalization()(x01)
    x01 = LeakyReLU(0.01)(x01)
    x01 = Dropout(0.2)(x01)

    x20 = conv2d(filters=int(64 * number_of_filters))(p1)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    x20 = Dropout(0.2)(x20)
    x20 = conv2d(filters=int(64 * number_of_filters))(x20)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    x20 = Dropout(0.2)(x20)
    p2 = MaxPooling2D(pool_size=(2, 2))(x20)

    x11 = conv2dtranspose(int(16 * number_of_filters))(x20)
    x11 = concatenate([x10, x11])
    x11 = conv2d(filters=int(16 * number_of_filters))(x11)
    x11 = BatchNormalization()(x11)
    x11 = LeakyReLU(0.01)(x11)
    x11 = conv2d(filters=int(16 * number_of_filters))(x11)
    x11 = BatchNormalization()(x11)
    x11 = LeakyReLU(0.01)(x11)
    x11 = Dropout(0.2)(x11)

    x02 = conv2dtranspose(int(16 * number_of_filters))(x11)
    x02 = concatenate([x00, x01, x02])
    x02 = conv2d(filters=int(16 * number_of_filters))(x02)
    x02 = BatchNormalization()(x02)
    x02 = LeakyReLU(0.01)(x02)
    x02 = conv2d(filters=int(16 * number_of_filters))(x02)
    x02 = BatchNormalization()(x02)
    x02 = LeakyReLU(0.01)(x02)
    x02 = Dropout(0.2)(x02)

    x30 = conv2d(filters=int(128 * number_of_filters))(p2)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    x30 = Dropout(0.2)(x30)
    x30 = conv2d(filters=int(128 * number_of_filters))(x30)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    x30 = Dropout(0.2)(x30)
    p3 = MaxPooling2D(pool_size=(2, 2))(x30)

    x21 = conv2dtranspose(int(16 * number_of_filters))(x30)
    x21 = concatenate([x20, x21])
    x21 = conv2d(filters=int(16 * number_of_filters))(x21)
    x21 = BatchNormalization()(x21)
    x21 = LeakyReLU(0.01)(x21)
    x21 = conv2d(filters=int(16 * number_of_filters))(x21)
    x21 = BatchNormalization()(x21)
    x21 = LeakyReLU(0.01)(x21)
    x21 = Dropout(0.2)(x21)

    x12 = conv2dtranspose(int(16 * number_of_filters))(x21)
    x12 = concatenate([x10, x11, x12])
    x12 = conv2d(filters=int(16 * number_of_filters))(x12)
    x12 = BatchNormalization()(x12)
    x12 = LeakyReLU(0.01)(x12)
    x12 = conv2d(filters=int(16 * number_of_filters))(x12)
    x12 = BatchNormalization()(x12)
    x12 = LeakyReLU(0.01)(x12)
    x12 = Dropout(0.2)(x12)

    x03 = conv2dtranspose(int(16 * number_of_filters))(x12)
    x03 = concatenate([x00, x01, x02, x03])
    x03 = conv2d(filters=int(16 * number_of_filters))(x03)
    x03 = BatchNormalization()(x03)
    x03 = LeakyReLU(0.01)(x03)
    x03 = conv2d(filters=int(16 * number_of_filters))(x03)
    x03 = BatchNormalization()(x03)
    x03 = LeakyReLU(0.01)(x03)
    x03 = Dropout(0.2)(x03)

    m = conv2d(filters=int(256 * number_of_filters))(p3)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    m = conv2d(filters=int(256 * number_of_filters))(m)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    m = Dropout(0.2)(m)

    x31 = conv2dtranspose(int(128 * number_of_filters))(m)
    x31 = concatenate([x31, x30])
    x31 = conv2d(filters=int(128 * number_of_filters))(x31)
    x31 = BatchNormalization()(x31)
    x31 = LeakyReLU(0.01)(x31)
    x31 = conv2d(filters=int(128 * number_of_filters))(x31)
    x31 = BatchNormalization()(x31)
    x31 = LeakyReLU(0.01)(x31)
    x31 = Dropout(0.2)(x31)

    x22 = conv2dtranspose(int(64 * number_of_filters))(x31)
    x22 = concatenate([x22, x20, x21])
    x22 = conv2d(filters=int(64 * number_of_filters))(x22)
    x22 = BatchNormalization()(x22)
    x22 = LeakyReLU(0.01)(x22)
    x22 = conv2d(filters=int(64 * number_of_filters))(x22)
    x22 = BatchNormalization()(x22)
    x22 = LeakyReLU(0.01)(x22)
    x22 = Dropout(0.2)(x22)

    x13 = conv2dtranspose(int(32 * number_of_filters))(x22)
    x13 = concatenate([x13, x10, x11, x12])
    x13 = conv2d(filters=int(32 * number_of_filters))(x13)
    x13 = BatchNormalization()(x13)
    x13 = LeakyReLU(0.01)(x13)
    x13 = conv2d(filters=int(32 * number_of_filters))(x13)
    x13 = BatchNormalization()(x13)
    x13 = LeakyReLU(0.01)(x13)
    x13 = Dropout(0.2)(x13)

    x04 = conv2dtranspose(int(16 * number_of_filters))(x13)
    x04 = concatenate([x04, x00, x01, x02, x03], axis=3)
    x04 = conv2d(filters=int(16 * number_of_filters))(x04)
    x04 = BatchNormalization()(x04)
    x04 = LeakyReLU(0.01)(x04)
    x04 = conv2d(filters=int(16 * number_of_filters))(x04)
    x04 = BatchNormalization()(x04)
    x04 = LeakyReLU(0.01)(x04)
    x04 = Dropout(0.2)(x04)

    output = Conv2D(num_classes, kernel_size=(1, 1), activation='sigmoid')(x04)

    # model = Model(inputs=inputs, outputs=output)
    # model.compile(optimizer=Adam(lr=0.0005), loss=weighted_loss(cce_iou_dice, weights_list), metrics=[dice, iou])
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=[iou, dice])


    if weights_input:
        model.load_weights(weights_input)

    return model

def prepare_input(image):
    image = np.reshape(image, image.shape+(1,))
    image = np.reshape(image,(1,)+image.shape)
    image = np.clip(image, 0, 255)
    return np.divide(image, 255)

def prepare_output(image):
    image = image[:,:,0]
    image = np.clip(image, 0, 1)
    return np.multiply(image, 255)