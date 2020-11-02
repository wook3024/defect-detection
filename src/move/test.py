import os
import glob
import shutil


# folder_list = glob.glob('../dataset/Class*/')
# folder_list = sorted(folder_list, key=lambda path: int(path.split('/')[2][5:]))
# folder_list = list(map(lambda path: path + 'train/aug', folder_list))
# remove_list = []
# for i in range(len(folder_list)):
#     if (i+1) % 2 == 0:
#         remove_list.append(folder_list[i])
# for path in remove_list:
#     print(path, os.path.exists(path))

# print(folder_list)
# print(os.getcwd())

save_folders = ['unet_efn_ori', 'unet_efn_IDG']

for save_folder in save_folders:
    print(save_folder)