import numpy as np
import cv2
import time
import socket
import io
from PIL import Image
import math

SERVER_IP = '192.xxx.xxx.xxx'
SERVER_PORT = 12397
SERVER_ADDR = (SERVER_IP, SERVER_PORT)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


# if cv2.__version__.startswith('2'):
#     PROP_FRAME_WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
#     PROP_FRAME_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
# elif cv2.__version__.startswith('3'):
#     PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
#     PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
# else:
#     PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
#     PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(PROP_FRAME_WIDTH, 256)
# cap.set(PROP_FRAME_HEIGHT, 256)

import threading
import cv2

class ThreadedCapture:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

# cap = cv2.VideoCapture(0)
cap = ThreadedCapture(0)
cap.start()

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

size = (256, 256)


# videoFile1 = './accuracy_check_cutting.mp4'
# cap_video = cv2.VideoCapture(videoFile1)

# out = cv2.VideoWriter(
#     'qwer222.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)


start = 0
fps_set = []
for i in range(10):
    fps_set.append(20)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect(SERVER_ADDR)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(256,256), interpolation=cv2.INTER_AREA)
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        stringData = np.array(imgencode).tobytes()
        client_socket.send(str(len(stringData)).ljust(16).encode())
        client_socket.send(stringData)

        length = recvall(client_socket, 16)
        stringData = recvall(client_socket, int(length))
        data = np.fromstring(stringData, dtype='uint8')
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # frame = cv2.resize(frame, dsize=(256,256), interpolation=cv2.INTER_AREA)

        fps = math.ceil(1/(time.time()-start+1e-10))
        
        start = time.time()

        fps_set.append(fps)
        fps_set.pop(0)
        
        cv2.putText(frame, f"fps:{np.mean(fps_set)}", (0, 100),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 3)
        # out.write(frame)
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.stop()
            break
