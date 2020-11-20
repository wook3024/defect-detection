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
# from swa.tfkeras import SWA
from skimage.io import imread
from skimage.transform import resize
import math
import cv2


plt.style.use('seaborn-white')
sns.set_style("white")


class NeuralNetwork():
    def __init__(self):
        self.arch = importlib.import_module(
            "%s.%s.%s" % (const.dn_NN, const.dn_ARCH, const.MODEL))

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
            sys.exit("\nError loading: %s\n%s\n" %
                     (self.fn_checkpoint, str(e)))

    def has_checkpoint(self):
        return self.fn_checkpoint if path.exist(self.fn_checkpoint) else None

    def prepare_data(self, images, labels=None):
        while True:
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

    def prepare_one_data(self, image, labels=None):
        # print(image.shape)
        image, _ = dip.preprocessor(image, None)
        # print(image.shape)
        image = self.arch.prepare_input(image)
        # print(image.shape)
        return image


    def overlay_data(self, original, image):
        image = dip.posprocessor(
            original, self.arch.prepare_output(image))
        # print("dip.pospro", image.shape)
        image = dip.overlay(original, image)
        return image

    def save_predict(self, original, image):
        path_save = path.join(self.dn_test_out, mkdir=True)

        with open(path.join(path_save, (const.fn_SEGMENTATION)), 'w+') as f:
            for (i, image) in enumerate(image):
                number = ("%0.3d" % (i+1))
                image_name = (const.fn_PREDICT % (number))

                original_name = (const.fn_ORIGINAL % (number))
                data.imwrite(path.join(path_save, original_name), original[i])
                
                print("posprocessor shape", image.shape, original[i].shape, self.arch.prepare_output(image).shape)
                image = dip.posprocessor(
                    original[i], self.arch.prepare_output(image))
                print("posprocessor shape", image.shape)
                data.imwrite(path.join(path_save, image_name), image)

                seg = (image == 255).sum()
                f.write(("Image %s was approximately %f segmented (%s pixels)\n" % (
                    number, (seg/image.size), seg)))

                overlay_name = (const.fn_OVERLAY % (number))
                overlay = dip.overlay(original[i], image)
                data.imwrite(path.join(path_save, overlay_name), overlay)
        f.close()


def train():
    nn = NeuralNetwork()

    total = data.length_from_path(nn.dn_image, nn.dn_aug_image)
    # q = misc.round_up(total, 100) - total
    q = total * 3

    gen.augmentation()
    # if (const.fn_cur_count == 1):
    #     print("Dataset augmentation (%s increase) is necessary (only once)\n" % q)
    #     if (const.save_folder == 'not_unet_efn_IDG'):
            # gen.augmentation(q)
            # gen.augmentation()

    images, labels = data.fetch_from_paths([nn.dn_image, nn.dn_aug_image], [
                                           nn.dn_label, nn.dn_aug_label])
    
    print(f"Image size: {len(images)}")

    images, labels, v_images, v_labels = misc.random_split_dataset(
        images, labels, const.p_VALIDATION)

    print(f"Train size: {len(images)} \nValidation size: {len(v_images)}")

    class crackSequence(tf.keras.utils.Sequence):
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) *
            self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) *
            self.batch_size]

            # np.array([
            #     resize(imread(file_name), (256, 256))
            #     for file_name in batch_x]), np.array(batch_y)
            # print(imread(batch_x[0]).shape)
            # y_val = resize(imread(batch_y[0]), (256, 256))
            # print("y_val###############", y_val.shape)
            # y_val = np.float32(y_val)
            # y_val = cv2.cvtColor(y_val, cv2.COLOR_BGRA2BGR)
            # print("y_val###############", y_val.shape)
            
            # print(cv2.cvtColor(resize(imread(batch_y[0]), (256, 256)), cv2.COLOR_RGBA2RGB))
            X = np.array([resize(imread(file_name), (256, 256)) for file_name in batch_x])
            y = np.array([cv2.cvtColor(np.float32(resize(imread(file_name), (256, 256))), cv2.COLOR_BGRA2BGR) for file_name in batch_y])
            # print(X.shape)
            # print(y.shape)
            return X, y

    train_dataloader = crackSequence(images, labels, 4)
    val_dataloader = crackSequence(v_images, v_labels, 4)
    

    # epochs, steps_per_epoch, validation_steps = misc.epochs_and_steps(
    #     len(images), len(v_images))

    # print("Train size:\t\t%s |\tSteps per epoch: \t%s\nValidation size:\t%s |\tValidation steps:\t%s\n"
    #       % misc.str_center(len(images), steps_per_epoch, len(v_images), validation_steps))
    # print(f"Train size: {len(images)} \nValidation size: {len(v_images)}")
    # print(train_dataloader.__getitem__(0)[0].shape)

    patience, patience_early = const.PATIENCE, int(len(train_dataloader)*0.25)
    loop, past_monitor = 0, float('inf')

    checkpoint = ModelCheckpoint(
        nn.fn_checkpoint, monitor=const.MONITOR, mode='min', save_best_only=True, verbose=2)
    early_stopping = EarlyStopping(monitor=const.MONITOR, mode='min', min_delta=const.MIN_DELTA,
                                   patience=const.PATIENCE, restore_best_weights=True, verbose=2)
    logger = CSVLogger(nn.fn_logger, append=True)

    h_iou = []
    h_val_iou = []
    h_loss = []
    h_val_loss = []
    while True:
        loop += 1
        # h = nn.model.fit_generator(
        #     shuffle=True,
        #     generator=nn.prepare_data(images, labels),
        #     steps_per_epoch=steps_per_epoch,
        #     epochs=1,
        #     validation_steps=validation_steps,
        #     validation_data=nn.prepare_data(v_images, v_labels),
        #     use_multiprocessing=True,
        #     callbacks=[checkpoint, early_stopping, logger])
        h = nn.model.fit_generator(
            shuffle=True,# python main.py --dataset=re_training_dataset --arch=unet+efn --dip=simple --gpu --camera

            generator=train_dataloader,
            epochs=70,
            validation_data=val_dataloader,
            # use_multiprocessing=True,
            callbacks=[checkpoint, logger])
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
        plt.savefig(f"fig/{const.dn_NN}:{const.dn_ARCH}:{const.MODEL}:{const.DATASET}_iou_{const.fn_cur_count}", dpi=300)
        # plt.show()
        plt.clf()

        # 7 visualization(loss)
        plt.plot(h.history['loss'], color='blue')
        plt.plot(h.history['val_loss'], color='red')
        plt.title('Model val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f"fig/{const.dn_NN}:{const.dn_ARCH}:{const.MODEL}:{const.DATASET}_loss_{const.fn_cur_count}", dpi=300)
        # plt.show()
        break

        h_iou.append(h.history['iou'][0])
        h_val_iou.append(h.history['val_iou'][0])
        h_loss.append(h.history['loss'][0])
        h_val_loss.append(h.history['val_iou'][0])

        if ("loss" in const.MONITOR):
            val_monitor = min(val_monitor)
            improve = (past_monitor - val_monitor)
        else:
            val_monitor = max(val_monitor)
            improve = (val_monitor - past_monitor)

        print("##################")
        print("Finished epoch (%s) with %s: %f" %
              (loop, const.MONITOR, val_monitor))

        test(nn)
        if (abs(improve) == float("inf") or improve > const.MIN_DELTA):
            print("Improved from %f to %f" % (past_monitor, val_monitor))
            past_monitor = val_monitor
            patience = const.PATIENCE
        elif (patience > 0):
            print("Did not improve from %f" % (past_monitor))
            print("Current patience: %s" % (patience))
            patience -= 1
        else:
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


def test(nn=None):
    if nn is None:
        nn = NeuralNetwork()
    
    if (nn.has_checkpoint()):
        images = data.fetch_from_path(nn.dn_test)
        print(images.shape)
        generator = nn.prepare_data(images)

        while True:
            results = nn.model.predict_generator(generator, len(images), verbose=1)
        # nn.save_predict(images, results)
    else:
        print(">> Model not found (%s)\n" % nn.fn_checkpoint)
    

def camera(nn=None):
    if nn is None:
        nn = NeuralNetwork()

    if (nn.has_checkpoint()):
        if cv2.__version__.startswith('2'):
            PROP_FRAME_WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
            PROP_FRAME_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
        elif cv2.__version__.startswith('3'):
            PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
            PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        else:
            PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
            PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT

        #cap = cv2.VideoCapture(1)
        cap = cv2.VideoCapture('iphone-video_vuyKtczu_eYj2.mp4')
        # cap.set(PROP_FRAME_WIDTH, 1980)
        # cap.set(PROP_FRAME_HEIGHT, 1080)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        size = (frame_width, frame_height)

        out = cv2.VideoWriter(
            'output_iphone_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

        count = 0
        start = 0
        count_set = np.array([])

        ret, frame = cap.read()
        while ret:

            # if ret == True:
            #     frame = cv2.flip(frame, 0)

            # fps = 1/(time.time()-start+1e-10)
            # start = time.time()
            # print(f"Estimated frames per second : {fps}")
            print(count)
            count += 1
            # cv2.putText(frame, f"fps:{fps}", (0, 100),
            #             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0))
            # count_set = np.append(count_set, np.array([fps]))
            # if count >= 20:
            #     # fps = np.mean(count_set)
            #     # str = f"FPS : {fps}"
            #     # cv2.putText(frame, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            #     count = 0
            #     print(f"Estimated frames per second : {np.mean(count_set)}")
            #     count_set = np.array([])

            original_frame = frame
            # print(frame.shape)
            # frame = np.expand_dims(frame, axis=0)
            frame = nn.prepare_one_data(frame)
            # print(frame.shape)
            frame = np.squeeze(nn.model.predict(frame), axis=0)
            # print(frame.shape)
            frame = nn.overlay_data(original_frame, frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            out.write(frame)
            # nn.save_predict(original_frame, frame)
            # cv2.imshow("preview", frame)
            # count+=1
            ret, frame = cap.read()
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # while True:
        # results = nn.model.predict_generator(generator, len(images), verbose=1)
        # nn.save_predict(images, results)
    else:
        print(">> Model not found (%s)\n" % nn.fn_checkpoint)



import io
import json

# from torchvision import models 
# import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def server_build(nn=None):
    app = Flask(__name__)

    if nn is None:
        nn = NeuralNetwork()
    
    # imagenet_class_index = json.load(open('./imagenet_class_index.json'))
    # model = models.densenet121(pretrained=True)
    # model.eval()


    # def transform_image(image_bytes):
    #     my_transforms = transforms.Compose([transforms.Resize(255),
    #                                         transforms.CenterCrop(224),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize(
    #                                             [0.485, 0.456, 0.406],
    #                                             [0.229, 0.224, 0.224]
    #                                         )])
    #     image = Image.open(io.BytesIO(image_bytes))
    #     return my_transforms(image).unsqueeze(0)

    def get_prediction(image_bytes):
        # print(type(image_bytes),  len(image_bytes))
        # encoded_img = np.fromstring(image_bytes, dtype=np.uint8)
        # frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        frame = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = np.reshape(frame, (640, 480, 3))
        # print(type(frame))
        # print(frame.shape)
        # frame = cv2.resize(frame, dsize=(640, 480,), interpolation=cv2.INTER_AREA)
        # print(frame.shape)
        original_frame = frame
        # print(frame.shape)
        # frame = np.expand_dims(frame, axis=0)
        frame = nn.prepare_one_data(frame)
        # print(frame.shape)
        frame = nn.model.predict(frame)
        frame = np.squeeze(frame, axis=0)
        # print(frame.shape)
        frame = nn.overlay_data(original_frame, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # tensor = transform_image(image_bytes=image_bytes)
        # outputs = model.forward(tensor)
        # _, y_hat = outputs.max(1)
        # predicted_idx = str(y_hat.item())
        # imagenet_class_index[predicted_idx]
        return frame

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            file = request.files['file']
            img_bytes = file.read()
            # class_id, class_name = get_prediction(image_bytes=img_bytes)
            image = get_prediction(image_bytes=img_bytes).tolist()
            return jsonify({"image": image})
        # return jsonify("llll")
        # return json.dumps({"image": image}, cls=NumpyEncoder)

    # if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, threaded=False)


import socket

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def socket_image(nn=None):
    if nn is None:
        nn = NeuralNetwork()
    # 통신 정보 설정
    IP = '0.0.0.0'
    PORT = 5050
    SIZE = 480*480*3
    ADDR = (IP, PORT)
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

    # 서버 소켓 설정
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(ADDR)  # 주소 바인딩
        server_socket.listen()  # 클라이언트의 요청을 받을 준비

        # 무한루프 진입
        while True:
            client_socket, client_addr = server_socket.accept()  # 수신대기, 접속한 클라이언트 정보 (소켓, 주소) 반환
            while True:
                length = recvall(client_socket,16)
                stringData = recvall(client_socket, int(length))
                data = np.fromstring(stringData, dtype='uint8')
                frame = cv2.imdecode(data, 1)
                # print(frame.shape)
                # frame = client_socket.recv(SIZE)  # 클라이언트가 보낸 메시지 반환
                # frame = np.frombuffer(frame, dtype=np.uint8)
                # frame = np.reshape(frame, (640, 480, 3))
                original_frame = frame
                frame = nn.prepare_one_data(frame)
                frame = nn.model.predict(frame)
                frame = np.squeeze(frame, axis=0)
                frame = nn.overlay_data(original_frame, frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                
                result, imgencode = cv2.imencode('.jpg', frame, encode_param)
                stringData = np.array(imgencode).tobytes()

                client_socket.send(str(len(stringData)).ljust(16).encode())
                client_socket.send(stringData)
                # client_socket.send(frame)  # 클라이언트에게 응답

            client_socket.close()  # 클라이언트 소켓 종료