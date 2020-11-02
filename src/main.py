import tensorflow as tf
import argparse
from setting import environment, constant
from util import path, generator
from nn import nn
<<<<<<< HEAD
import os


# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
# 	print("Name:", gpu.name, " Type:", gpu.device_type)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

=======


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	print("Name:", gpu.name, " Type:", gpu.device_type)

>>>>>>> ae1365a80e67c67296f5c96f496887e49b4867b3
### gdown https://drive.google.com/uc?id=0B7EVK8r0v71pOXBhSUdJWU1MYUk
### python main.py --dip=example --tolabel
### python main.py --dataset=example --dip=example --augmentation=0000
### python main.py --dataset=example --arch=example --dip=example --gpu --test
### python main.py --dataset=example --arch=example --dip=example --gpu --train

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tolabel", help="Preprocess images to create labels (out/tolabel)", action="store_true", default=False)
	parser.add_argument("--augmentation", help="Dataset augmentation (pass quantity)", type=int)
	parser.add_argument("--dataset", help="Dataset name", type=str, default=constant.DATASET)
	parser.add_argument("--train", help="Train", action="store_true", default=False)
	parser.add_argument("--test", help="Predict", action="store_true", default=False)
	parser.add_argument("--arch", help="Neural Network architecture", type=str, default=constant.MODEL)
	parser.add_argument("--dip", help="Method for image processing", type=str, default=constant.IMG_PROCESSING)
	parser.add_argument("--gpu", help="Enable GPU mode", action="store_true", default=False)
<<<<<<< HEAD
	parser.add_argument("--count", help="cur_count", type=int, default=9)
=======
	parser.add_argument("--count", help="cur_count", type=int, default=7)
>>>>>>> ae1365a80e67c67296f5c96f496887e49b4867b3
	parser.add_argument("--save_folder", help="save_folder", type=str, default=constant.save_folder)
	args = parser.parse_args()

	environment.setup(args)
	exist = lambda x: len(x)>0 and path.exist(path.data(x, mkdir=False))

	if (args.tolabel):
		generator.tolabel()
	elif args.dataset is not None and exist(args.dataset):
		
		if (args.augmentation):
			generator.augmentation(args.augmentation)
		elif (args.train):
			nn.train()
		elif (args.test):
			nn.test()
	else:
		print("\n>> Dataset not found\n")

if __name__ == "__main__":
	main()