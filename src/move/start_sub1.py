import os
from pynvml import *
from time import sleep
import glob
import shutil

# os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
select_arch = 'unet+efn'
IMG_PROCESSING = 'simple'
# save_folder = 'not_unet_efn_Albu'
save_folders = ['unet_efn_IDG']

# flag = 0
# while flag < 5:
#     nvmlInit()
#     h = nvmlDeviceGetHandleByIndex(0)
#     info_1 = nvmlDeviceGetMemoryInfo(h)
#     if info_1.used <= 150:
#         flag += 1
#     else:
#         flag = 0
#     sleep(60)




for save_folder in save_folders:
    # folder_list = glob.glob('../dataset/Class*/')
    # folder_list = sorted(folder_list, key=lambda path: int(path.split('/')[2][5:]))
    # folder_list = list(map(lambda path: path + 'train/aug', folder_list))
    # remove_list = []
    # for i in range(len(folder_list)):
    #     if i % 2 == 0:
    #         remove_list.append(folder_list[i])
    # for path in remove_list:
    #     print(path, os.path.exists(path))
    #     shutil.rmtree(path)
    # if len(remove_list):
    #     continue

    test = 2
    count = 9
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info_1 = nvmlDeviceGetMemoryInfo(h)
    # h = nvmlDeviceGetHandleByIndex(1)
    # info_2 = nvmlDeviceGetMemoryInfo(h)
    if info_1.used <= 150:
        os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py --dataset=Class{count} --arch={select_arch} --dip={IMG_PROCESSING} --count={test} --save_folder={save_folder} --gpu --train')
        count += 2
    # elif info_2.used <= 150:
    #     os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py --dataset=Class{count} --arch={select_arch} --dip={IMG_PROCESSING} --count={test} --save_folder={save_folder} --gpu --train')
    #     count += 2
        
    sleep(60)
