import os
from pynvml import *
from time import sleep
import start_util

# os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

for test in range(1, 6):
    count = 1
    while count <= 10:
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info_1 = nvmlDeviceGetMemoryInfo(h)
        # h = nvmlDeviceGetHandleByIndex(1)
        # info_2 = nvmlDeviceGetMemoryInfo(h)
        if info_1.used <= 150:
            os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py --dataset=Class{count} --arch={start_util.select_arch} --dip={start_util.IMG_PROCESSING} --count={test} --save_folder={start_util.save_folder} --gpu --train')
            count += 2
        # elif info_2.used <= 150:
        #     os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py --dataset=Class{count} --arch={start_util.select_arch} --dip={start_util.IMG_PROCESSING} --count={test} --save_folder={start_util.save_folder} --gpu --train')
        #     count += 2
            
        sleep(60)
    