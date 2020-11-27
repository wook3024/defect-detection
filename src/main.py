import tensorflow as tf
import argparse
from setting import environment, constant
from util import path, generator
from nn import nn
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

### gdown https://drive.google.com/uc?id=0B7EVK8r0v71pOXBhSUdJWU1MYUk
### python main.py --dip=example --tolabel
### python main.py --dataset=example --dip=example --augmentation=0000
### python main.py --dataset=example --arch=example --dip=example --gpu --test
### python main.py --dataset=example --arch=example --dip=example --gpu --train

### python main.py --dataset=re_training_dataset --arch=unet+efn --dip=simple --gpu --train
# python main.py --dataset=re_training_dataset --arch=unet+efn --dip=simple --gpu --camera


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tolabel", help="Preprocess images to create labels (out/tolabel)", action="store_true", default=False)
	parser.add_argument("--augmentation", help="Dataset augmentation (pass quantity)", type=int)
	parser.add_argument("--dataset", help="Dataset name", type=str, default=constant.DATASET)
	parser.add_argument("--train", help="Train", action="store_true", default=False)
	parser.add_argument("--test", help="Predict", action="store_true", default=False)
	parser.add_argument("--camera", help="Predict", action="store_true", default=False)
	parser.add_argument("--server_build", help="Predict", action="store_true", default=False)
	parser.add_argument("--socket_image", help="Predict", action="store_true", default=False)
	parser.add_argument("--arch", help="Neural Network architecture", type=str, default=constant.MODEL)
	parser.add_argument("--dip", help="Method for image processing", type=str, default=constant.IMG_PROCESSING)
	parser.add_argument("--gpu", help="Enable GPU mode", action="store_true", default=False)
	parser.add_argument("--count", help="cur_count", type=int, default=9)
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
		elif (args.camera):
			nn.camera()
		elif (args.server_build):
			nn.server_build()
		elif (args.socket_image):
			nn.socket_image()
	else:
		print("\n>> Dataset not found\n")

if __name__ == "__main__":
	main()