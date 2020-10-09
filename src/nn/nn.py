<<<<<<< HEAD
from PIL import Image
import random
import os
import shutil
import glob
from skimage.transform import resize
import seaborn as sns
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.callbacks import Callback
from util import path, data, misc, generator as gen
from dip import dip
import keras
import keras.callbacks as callbacks
import setting.constant as const
import importlib
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from swa.tfkeras import SWA

plt.style.use('seaborn-white')
sns.set_style("white")


class NeuralNetwork():
    def __init__(self):
        self.arch = importlib.import_module(
            "%s.%s.%s" % (const.dn_NN, const.dn_ARCH, const.MODEL))
=======
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from util import path, data, misc, generator as gen
from dip import dip
import setting.constant as const
import importlib
import sys

class NeuralNetwork():
    def __init__(self): 
        self.arch = importlib.import_module("%s.%s.%s" % (const.dn_NN, const.dn_ARCH, const.MODEL))
>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1

        self.fn_logger = path.fn_logger()
        self.fn_checkpoint = path.fn_checkpoint()

        self.dn_image = path.dn_train(const.dn_IMAGE)
        self.dn_aug_image = path.dn_aug(const.dn_IMAGE, mkdir=False)

        self.dn_label = path.dn_train(const.dn_LABEL)
        self.dn_aug_label = path.dn_aug(const.dn_LABEL, mkdir=False)

        self.dn_test = path.dn_test()
        self.dn_test_out = path.dn_test(out_dir=True, mkdir=False)

        try:
            self.model = self.arch.model(self.has_checkpoint())
            if (self.has_checkpoint()):
                print("Loaded: %s\n" % self.fn_checkpoint)
        except Exception as e:
<<<<<<< HEAD
            sys.exit("\nError loading: %s\n%s\n" %
                     (self.fn_checkpoint, str(e)))
=======
            sys.exit("\nError loading: %s\n%s\n" % (self.fn_checkpoint, str(e)))
>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1

    def has_checkpoint(self):
        return self.fn_checkpoint if path.exist(self.fn_checkpoint) else None

    def prepare_data(self, images, labels=None):
<<<<<<< HEAD
        while True:
            if (labels is None):
                for (i, image) in enumerate(images):
                    number = ("%0.3d" % (i+1))
                    path_save = path.join(self.dn_test_out, mkdir=True)

                    image, _ = dip.preprocessor(image, None)
                    original_name = (const.fn_PREPROCESSING % (number))
                    # data.imwrite(path.join(path_save, original_name), image)

                    yield self.arch.prepare_input(image)
            else:
                for (image, label) in zip(images, labels):
                    (image, label) = dip.preprocessor(image, label)
                    yield self.arch.prepare_input(image), self.arch.prepare_input(label)
=======
        if (labels is None):
            for (i, image) in enumerate(images):
                number = ("%0.3d" % (i+1))
                path_save = path.join(self.dn_test_out, mkdir=True)

                image, _ = dip.preprocessor(image, None)
                original_name = (const.fn_PREPROCESSING % (number))
                data.imwrite(path.join(path_save, original_name), image)

                yield self.arch.prepare_input(image)
        else:
            for (image, label) in zip(images, labels):
                (image, label) = dip.preprocessor(image, label)
                yield self.arch.prepare_input(image), self.arch.prepare_input(label)
>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1

    def save_predict(self, original, image):
        path_save = path.join(self.dn_test_out, mkdir=True)

        with open(path.join(path_save, (const.fn_SEGMENTATION)), 'w+') as f:
            for (i, image) in enumerate(image):
                number = ("%0.3d" % (i+1))
                image_name = (const.fn_PREDICT % (number))

<<<<<<< HEAD
                original_name = (const.fn_ORIGINAL % (number))
                data.imwrite(path.join(path_save, original_name), original[i])

                image = dip.posprocessor(
                    original[i], self.arch.prepare_output(image))
                data.imwrite(path.join(path_save, image_name), image)

                seg = (image == 255).sum()
                f.write(("Image %s was approximately %f segmented (%s pixels)\n" % (
                    number, (seg/image.size), seg)))

                overlay_name = (const.fn_OVERLAY % (number))
                overlay = dip.overlay(original[i], image)
                # data.imwrite(path.join(path_save, overlay_name), overlay)
        f.close()


=======
                image = dip.posprocessor(original[i], self.arch.prepare_output(image))
                data.imwrite(path.join(path_save, image_name), image)

                seg = (image == 255).sum()
                f.write(("Image %s was approximately %f segmented (%s pixels)\n" % (number, (seg/image.size), seg)))

                original_name = (const.fn_ORIGINAL % (number))
                data.imwrite(path.join(path_save, original_name), original[i])

                overlay_name = (const.fn_OVERLAY % (number))
                overlay = dip.overlay(original[i], image)
                data.imwrite(path.join(path_save, overlay_name), overlay)
        f.close()

>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1
def train():
    nn = NeuralNetwork()

    total = data.length_from_path(nn.dn_image, nn.dn_aug_image)
<<<<<<< HEAD
    # q = misc.round_up(total, 100) - total
    q = total * 3

    if (const.fn_cur_count == 1):
        print("Dataset augmentation (%s increase) is necessary (only once)\n" % q)
        if (const.save_folder == 'not_unet_efn_IDG'):
            gen.augmentation(q)
            # gen.augmentation()

    images, labels = data.fetch_from_paths([nn.dn_image, nn.dn_aug_image], [
                                           nn.dn_label, nn.dn_aug_label])
    images, labels, v_images, v_labels = misc.random_split_dataset(
        images, labels, const.p_VALIDATION)

    epochs, steps_per_epoch, validation_steps = misc.epochs_and_steps(
        len(images), len(v_images))

    print("Train size:\t\t%s |\tSteps per epoch: \t%s\nValidation size:\t%s |\tValidation steps:\t%s\n"
          % misc.str_center(len(images), steps_per_epoch, len(v_images), validation_steps))
    print("Epochs:\t\t", epochs)
=======
    q = misc.round_up(total, 100) - total

    if (q > 0):
        print("Dataset augmentation (%s increase) is necessary (only once)\n" % q)
        gen.augmentation(q)

    images, labels = data.fetch_from_paths([nn.dn_image, nn.dn_aug_image], [nn.dn_label, nn.dn_aug_label])
    images, labels, v_images, v_labels = misc.random_split_dataset(images, labels, const.p_VALIDATION)
    
    epochs, steps_per_epoch, validation_steps = misc.epochs_and_steps(len(images), len(v_images))

    print("Train size:\t\t%s |\tSteps per epoch: \t%s\nValidation size:\t%s |\tValidation steps:\t%s\n" 
        % misc.str_center(len(images), steps_per_epoch, len(v_images), validation_steps))
>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1

    patience, patience_early = const.PATIENCE, int(epochs*0.25)
    loop, past_monitor = 0, float('inf')

<<<<<<< HEAD
    checkpoint = ModelCheckpoint(
        nn.fn_checkpoint, monitor=const.MONITOR, mode='min', save_best_only=True, verbose=2)
    early_stopping = EarlyStopping(monitor=const.MONITOR, mode='min', min_delta=const.MIN_DELTA,
                                   patience=const.PATIENCE, restore_best_weights=True, verbose=2)
    logger = CSVLogger(nn.fn_logger, append=True)

    h_iou = []
    h_val_iou = []
    h_loss = []
    h_val_loss = []
=======
    checkpoint = ModelCheckpoint(nn.fn_checkpoint, monitor=const.MONITOR, mode='max', save_best_only=True, verbose=1)
    #previous restore_best_weights is True
    #monitor에 iou를 넣어서 mode='max'사용
    early_stopping = EarlyStopping(monitor=const.MONITOR, mode='max', min_delta=const.MIN_DELTA, patience=patience_early, restore_best_weights=True, verbose=1)
    logger = CSVLogger(nn.fn_logger, append=True)

>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1
    while True:
        loop += 1
        h = nn.model.fit_generator(
            shuffle=True,
            generator=nn.prepare_data(images, labels),
            steps_per_epoch=steps_per_epoch,
<<<<<<< HEAD
            epochs=200,
=======
            epochs=epochs,
>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1
            validation_steps=validation_steps,
            validation_data=nn.prepare_data(v_images, v_labels),
            use_multiprocessing=True,
            callbacks=[checkpoint, early_stopping, logger])
<<<<<<< HEAD
        # [checkpoint, early_stopping, logger]
        val_monitor = h.history[const.MONITOR]

        test(nn)
        # print(h.history['val_iou'])
        generator = nn.prepare_data(v_images, v_labels)
        results = nn.model.evaluate_generator(generator, steps=1)
        # print("results : ", results)

        # visualization(iou)
        plt.plot(h.history['iou'])
        plt.plot(h.history['val_iou'])
        plt.title('Model val_iou')
        plt.xlabel('Epoch')
        plt.ylabel('Iou')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f"test{const.fn_cur_count}/{const.save_folder}/{const.dn_NN}:{const.dn_ARCH}:{const.MODEL}:{const.DATASET}_iou_{const.fn_cur_count}", dpi=300)
        # plt.show()
        plt.clf()

        # 7 visualization(loss)
        plt.plot(h.history['loss'], color='blue')
        plt.plot(h.history['val_loss'], color='red')
        plt.title('Model val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f"test{const.fn_cur_count}/{const.save_folder}/{const.dn_NN}:{const.dn_ARCH}:{const.MODEL}:{const.DATASET}_loss_{const.fn_cur_count}", dpi=300)
        # plt.show()
        break

        h_iou.append(h.history['iou'][0])
        h_val_iou.append(h.history['val_iou'][0])
        h_loss.append(h.history['loss'][0])
        h_val_loss.append(h.history['val_iou'][0])

=======

        val_monitor = h.history[const.MONITOR]
        
>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1
        if ("loss" in const.MONITOR):
            val_monitor = min(val_monitor)
            improve = (past_monitor - val_monitor)
        else:
            val_monitor = max(val_monitor)
            improve = (val_monitor - past_monitor)

<<<<<<< HEAD
        print("##################")
        print("Finished epoch (%s) with %s: %f" %
              (loop, const.MONITOR, val_monitor))

        # test(nn)
=======
        print("\n##################")
        print("Finished epoch (%s) with %s: %f" % (loop, const.MONITOR, val_monitor))

>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1
        if (abs(improve) == float("inf") or improve > const.MIN_DELTA):
            print("Improved from %f to %f" % (past_monitor, val_monitor))
            past_monitor = val_monitor
            patience = const.PATIENCE
<<<<<<< HEAD
=======
            test(nn)
>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1
        elif (patience > 0):
            print("Did not improve from %f" % (past_monitor))
            print("Current patience: %s" % (patience))
            patience -= 1
        else:
<<<<<<< HEAD
            test(nn)
            plt.plot(h_iou)
            plt.plot(h_val_iou)
            plt.title('Model val_iou')
            plt.xlabel('Epoch')
            plt.ylabel('Iou')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig(
                f"fig/{const.dn_NN}:{const.dn_ARCH}:{const.MODEL}:{const.DATASET}_iou_{const.fn_cur_count}", dpi=300)
            plt.show()

            # 7 훈련 과정 시각화 (손실)
            plt.plot(h_loss)
            plt.plot(h_val_loss)
            plt.title('Model val_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig(
                f"fig/{const.dn_NN}:{const.dn_ARCH}:{const.MODEL}:{const.DATASET}_loss_{const.fn_cur_count}", dpi=300)
            plt.show()
            break
        print("##################")

=======
            break
        print("##################\n")
>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1

def test(nn=None):
    if nn is None:
        nn = NeuralNetwork()
<<<<<<< HEAD
    
=======

>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1
    if (nn.has_checkpoint()):
        images = data.fetch_from_path(nn.dn_test)
        generator = nn.prepare_data(images)

        results = nn.model.predict_generator(generator, len(images), verbose=1)
        nn.save_predict(images, results)
    else:
<<<<<<< HEAD
        print(">> Model not found (%s)\n" % nn.fn_checkpoint)
    
=======
        print(">> Model not found (%s)\n" % nn.fn_checkpoint)
>>>>>>> a703cd23bf51f3c2805f3fe4b435b1fef35e88b1
