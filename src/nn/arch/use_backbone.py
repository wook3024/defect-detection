from .UNetPlusPlus.segmentation_models import Unet, Nestnet, Xnet
import numpy as np

IMAGE_SIZE = (256,256,3)

def model(weights_input=None):
    # model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build UNet++
    model = Unet(backbone_name='resnet50', input_shape=IMAGE_SIZE, encoder_weights='imagenet', decoder_block_type='transpose') # build U-Net
    # model = NestNet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build DLA

    model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])
    
    model.summary()
    # if weights_input:
    #     model.load_weights(weights_input)

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