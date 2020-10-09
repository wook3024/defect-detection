import efficientnet.keras as efn
import tensorflow as tf
tf.random.Generator = None
import tensorflow_addons as tfa
import keras.backend as K
import keras.callbacks as callbacks
import numpy as np
import random
import os
from skimage.transform import resize
from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import  ModelCheckpoint
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Activation
from keras.layers import Add, concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm_notebook
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.losses import binary_crossentropy
from keras.callbacks import Callback
from keras.applications.xception import Xception
from keras.layers import multiply
from keras.optimizers import Adam, SGD

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

IMAGE_SIZE = (256,256,3)

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

@tf.function
def get_iou_vector(A, B):
    # Numpy version    
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = K.greater(t, 0)
        pred = K.greater(p, 0)
        
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels
        intersection = K.greaterum(t * p, 0)
        union = true + pred - intersection
        iou = intersection / union
        
        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = K.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric

@tf.function
def my_iou_metric(label, pred):
    return tf.py_function(get_iou_vector, [label, pred > 0.5], tf.float64)

@tf.function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

@tf.function
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

@tf.function
def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

@tf.function
def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x



def UEfficientNet(input_shape=IMAGE_SIZE, backbone_size='b0', pre_trained_weight='imagenet', dropout_rate=0.1):
    if backbone_size == 'b7':
        backbone = efn.EfficientNetB7(weights=pre_trained_weight, include_top=False, input_shape=input_shape)
        backbone_layer = [554, 258, 155, 52]
    elif backbone_size == 'b4':
        backbone = efn.EfficientNetB4(weights=pre_trained_weight, include_top=False, input_shape=input_shape)
        backbone_layer = [342, 154, 92, 30]
    else:
        backbone = efn.EfficientNetB0(weights=pre_trained_weight, include_top=False, input_shape=input_shape)
        backbone_layer = [158, 72, 44, 16]

    input = backbone.input
    start_neurons = 8

    # # backbone_layer check
    # for i, layer in enumerate(backbone.layers):
    #     print(i, layer.output)

    conv4 = backbone.layers[backbone_layer[0]].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)
    
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same",name='conv_middle')(pool4)
    convm = residual_block(convm,start_neurons * 32)
    convm = residual_block(convm,start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    # uconv4 = concatenate([deconv4, conv4])
    # uconv4 = Dropout(dropout_rate)(uconv4) 
    uconv4 = Dropout(dropout_rate)(deconv4) 
    
    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)  

    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    # conv3 = backbone.layers[backbone_layer[1]].output
    # uconv3 = concatenate([deconv3, conv3])    
    # uconv3 = Dropout(dropout_rate)(uconv3)
    uconv3 = Dropout(dropout_rate)(deconv3)
    
    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    # conv2 = backbone.layers[backbone_layer[2]].output
    # uconv2 = concatenate([deconv2, conv2])
        
    # uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Dropout(0.1)(deconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    # conv1 = backbone.layers[backbone_layer[3]].output
    # uconv1 = concatenate([deconv1, conv1])
    
    # uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Dropout(0.1)(deconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = Dropout(dropout_rate/2)(uconv0)
    #### bacause shape error, channel 1 -> 3  change ####
    output_layer = Conv2D(3, (1,1), padding="same", activation="sigmoid")(uconv0)    
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():  
        model = Model(input, output_layer)
        model.name = 'u-efficient'

    return model


def model(weights_input=None):
    model = UEfficientNet(input_shape=IMAGE_SIZE, backbone_size='b0', pre_trained_weight='imagenet', dropout_rate=0.5)

    opt = tfa.optimizers.SWA(
        tf.keras.optimizers.SGD(lr=2.0), 10, 3)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=[iou])

    # model.summary()
    
    if weights_input:
        #./keras_swa.model
        print("check weights_input ***", weights_input)
        try:
            model.load_weights("./model/keras_swa.hdf5")
            print('using swa weight model')
        except Exception as e:
            model.load_weights(weights_input)
            print(e)

    return model


def prepare_input(image):
    image = np.reshape(image, image.shape+(1,))
    image = np.reshape(image,(1,)+image.shape)
    image = np.squeeze(image, axis=-1)
    image = np.clip(image, 0, 255)
    # print(image.shape)
    return np.divide(image, 255)

def prepare_output(image):
    image = image[:,:,0]
    image = np.clip(image, 0, 1)
    # print(image.shape)
    return np.multiply(image, 255)