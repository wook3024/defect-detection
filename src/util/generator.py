from keras.preprocessing.image import ImageDataGenerator
from util import path, data
from dip import dip, image as im
import setting.constant as const
import numpy as np

import tensorflow as tf
import numpy as np
import keras

from skimage.transform import resize

import glob
import shutil
import os, os.path
import random
import matplotlib.pyplot as plt
from PIL import Image

import cv2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop, PadIfNeeded, VerticalFlip, RandomRotate90
)

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
# tf.set_random_seed(seed)
# tf.random.set_seed(seed)



def augmentation(multiple=3):
    print('my augmectation multiple: ', multiple)
    train_path = path.data(const.DATASET, const.dn_TRAIN)

    image_folder = image_save_prefix = const.dn_IMAGE
    label_folder = label_save_prefix = const.dn_LABEL

    image_to_dir = path.dn_aug(const.dn_IMAGE)
    label_to_dir = path.dn_aug(const.dn_LABEL)

    train_im_path = '/'.join([train_path, image_folder])
    train_mask_path = '/'.join([train_path, label_folder])

    h,w,batch_size = 256,256,16
    class DataGenerator(keras.utils.Sequence):
        'Generates data for Keras'
        def __init__(self, train_im_path=train_im_path,train_mask_path=train_mask_path,
                    augmentations=None, batch_size=batch_size,img_size=256, n_channels=3, shuffle=False):
            'Initialization'
            self.batch_size = batch_size
            self.train_im_paths = glob.glob(train_im_path+'/*')
            self.train_im_path = train_im_path
            self.train_mask_path = train_mask_path

            self.img_size = img_size
            
            self.n_channels = n_channels
            self.shuffle = shuffle
            self.augment = augmentations
            self.on_epoch_end()

        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.ceil(len(self.train_im_paths) / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.train_im_paths))]

            # Find list of IDs
            list_IDs_im = [self.train_im_paths[k] for k in indexes]

            # Generate data
            X, y = self.data_generation(list_IDs_im)

            if self.augment is None:
                return X,np.array(y)/255
            else:            
                im,mask = [],[]   
                for x,y in zip(X,y):
                    augmented = self.augment(image=x, mask=y)
                    im.append(augmented['image'])
                    mask.append(augmented['mask'])
                return np.array(im),np.array(mask)/255

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(len(self.train_im_paths))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

        def data_generation(self, list_IDs_im):
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            X = np.empty((len(list_IDs_im),self.img_size,self.img_size, self.n_channels))
            y = np.empty((len(list_IDs_im),self.img_size,self.img_size, 1))

            # Generate data
            for i, im_path in enumerate(list_IDs_im):
                
                im = np.array(Image.open(im_path))
                mask_path = im_path.replace(self.train_im_path,self.train_mask_path)
                
                mask = np.array(Image.open(mask_path))
                
                
                if len(im.shape)==2:
                    im = np.repeat(im[...,None],3,2)
                    
                if self.n_channels == 3:
                    im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
                    X[i,] = cv2.resize(im,(self.img_size,self.img_size))
                else:
                    im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
                    im = np.expand_dims(im, axis=-1)
                    print(im.shape, X[i,].shape)
                    X[i,] = cv2.resize(im,(self.img_size,self.img_size)).reshape(self.img_size, self.img_size, -1)
                    
                if mask.shape[-1] == 3:
                    mask = np.zeros((self.img_size, self.img_size), dtype=int)
                mask = np.array(mask, dtype='uint8')
                
                y[i,] = cv2.resize(mask,(self.img_size,self.img_size)).reshape(self.img_size, self.img_size, -1)
                y[y>0] = 255

            return np.uint8(X),np.uint8(y)

    aug = Compose([
        OneOf([
            RandomContrast(),
            RandomGamma(),
            RandomBrightness(),
            ], p=0.3),

        RandomRotate90(p=0.5),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#             GridDistortion(p=0.5),
            OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=1),
        ], p=0.8)
        ])

    
    batch_size = len([name for name in os.listdir(train_im_path) if os.path.isfile(os.path.join(train_im_path, name))])
    a = DataGenerator(batch_size=batch_size, 
                    train_im_path = train_im_path,
                    train_mask_path = train_mask_path,
                    augmentations=aug,
                    img_size=const.IMAGE_SIZE[0], 
                    shuffle=False)
    

    num = 1
    for index in range(multiple):
        images, masks = a.__getitem__(0)
        seed = int(np.random.rand(1)*100)    
        print(''.join(['size: ', str(len(images))]))
        random.seed(seed)
        for i,(im, mask) in enumerate(zip(images,masks)):
            number = ("%0.3d" % (num))
            cv2.imwrite(f'{image_to_dir}/{number}.png', im)
            plt.imsave(f'{label_to_dir}/{number}.png', mask.squeeze(), cmap='gray')
            num = num + 1

    print(''.join(['total size: ', str(len(images) * multiple)]))

# def augmentation(n=1):
#     batch_size = 1
#     target_size = const.IMAGE_SIZE[:2]
#     seed = int(np.random.rand(1)*100)

#     train_path = path.data(const.DATASET, const.dn_TRAIN)

#     image_folder = image_save_prefix = const.dn_IMAGE
#     label_folder = label_save_prefix = const.dn_LABEL

#     image_to_dir = path.dn_aug(const.dn_IMAGE)
#     label_to_dir = path.dn_aug(const.dn_LABEL)
    
#     image_gen = label_gen = ImageDataGenerator(
#         rotation_range=0.2,
#         fill_mode="constant",
#         rescale = 1./255,
#         width_shift_range=0.05,
#         height_shift_range=0.05, 
#         channel_shift_range=0.05,
#         shear_range=0.05,
#         zoom_range=0.05,
#         vertical_flip=True,
#         horizontal_flip=True)

#     image_batch = image_gen.flow_from_directory(
#         directory = train_path,
#         classes = [image_folder],
#         target_size = target_size,
#         batch_size = batch_size,
#         save_to_dir = image_to_dir,
#         save_prefix = image_save_prefix,
#         seed = seed)

#     label_batch = label_gen.flow_from_directory(
#         directory = train_path,
#         classes = [label_folder],
#         target_size = target_size,
#         batch_size = batch_size,
#         save_to_dir = label_to_dir,
#         save_prefix = label_save_prefix,
#         seed = seed)

#     for i, (_,_)  in enumerate(zip(image_batch, label_batch)):
#         if (i >= n-1): break

def tolabel():
    dn_tolabel = path.out(const.dn_TOLABEL, mkdir=False)

    if path.exist(dn_tolabel):
        
        dir_save = path.out(const.dn_TOLABEL)
        images = data.fetch_from_path(dir_save)

        for (i, image) in enumerate(images):
            path_save = path.join(dir_save, "label", mkdir=True)
            file_name = ("%0.3d.png" % (i+1))
            file_save = path.join(path_save, file_name)

            img_pp, _ = dip.preprocessor(image, None)
            img_pp = image
            data.imwrite(file_save, img_pp)
    else:
    	print("\n>> Folder not found (%s)\n" % dn_tolabel)