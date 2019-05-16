import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3 
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from imgaug import augmenters as iaa
from keras.utils.np_utils import to_categorical
import random
import numpy as np
import pydicom
from keras import backend as K
from create_model import CreateModel
from PIL import Image
from keras import layers, Model
from keras.optimizers import Adam
import pandas as pd


def light_augmentation(img):
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)
	sometimes2 = lambda aug: iaa.Sometimes(0.2, aug)
	seq = iaa.Sequential([
		# sometimes(iaa.Crop(px=(0, 16))), 
		iaa.Fliplr(0.5), 
		iaa.Flipud(0.5), 
		sometimes(
			iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
		),
		sometimes(iaa.Affine(
			scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
			translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
			rotate=(-5, 5), # rotate by -45 to +45 degrees
			shear=(-5, 5), # shear by -16 to +16 degrees
		))
	])
	return seq.augment_image(img)









def fn_reader(d, path, size):
	if d["type"] == "real":
		img = Image.open(path + "/data/all_rgb/" + d["patientId"]+".png").resize(size)
	else:
		img = Image.open(path + "/result_gan/rsna_1/" + d["patientId"]).resize(size)

	img = light_augmentation(np.array(img))
	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["Target"], num_classes=2)]

def fn_reader_val(d, path, size):
	img = np.array(Image.open(path + "/" + d["patientId"]+".png").resize(size))
	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["Target"], num_classes=2)]



train = CreateModel(
	DenseNet121, batch_size=32, batch_size_val = 32, save_model="models/dataset-2-rsna/", name_model="DenseNet121_224x224_GAN_1",
	add_top=False, class_mode="categorical", freeze=None, epochs=50, dropout=True,
	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, 
	# weight=[0.64541409, 2.21922821],
	# load_model="models/dataset-4-skincancer/comparaison/Mobilnet_Resampling/best_model_epoch.hdf5"
)
train.set_train_generator(
	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2-rsna/result_gan/train_real_fake__01.csv",
	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2-rsna"
)
train.set_val_generator(
	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2-rsna/test.csv",
	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2-rsna/data/all_rgb"
)
train.run()




# ---------- Dataset - 2 --------------


# def fn_reader(d, path, size):
# 	img = Image.open(path + "/" + d["patientId"]+".png").resize(size)
# 	img = light_augmentation(np.array(img))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["Target"], num_classes=2)]

# def fn_reader_val(d, path, size):
# 	img = np.array(Image.open(path + "/" + d["patientId"]+".png").resize(size))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["Target"], num_classes=2)]





# train = CreateModel(
# 	DenseNet121, batch_size=32, batch_size_val = 32, save_model="models/dataset-2-rsna/", name_model="DenseNet121_224x224",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=50, dropout=True,
# 	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, 
# 	weight=[0.64541409, 2.21922821],
# 	# load_model="models/dataset-4-skincancer/comparaison/Mobilnet_Resampling/best_model_epoch.hdf5"
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2-rsna/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2-rsna/data/all_rgb"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2-rsna/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2-rsna/data/all_rgb"
# )
# train.run()



