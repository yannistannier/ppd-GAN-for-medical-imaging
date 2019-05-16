import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3 
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.applications.densenet import DenseNet121
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
		)),
		sometimes2(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200)))
	])
	return seq.augment_image(img)



# ---------- Dataset - 4 - skin Cancer -- Resampling --------------


def fn_reader(d, path, size):
	img = Image.open(path + "/" + d["image_id"]+".jpg").resize(size)
	img = light_augmentation(np.array(img))
	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label"], num_classes=7)]

def fn_reader_val(d, path, size):
	img = np.array(Image.open(path + "/" + d["image_id"]+".jpg").resize(size))
	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label"], num_classes=7)]





train = CreateModel(
	DenseNet121, batch_size=32, batch_size_val = 32, save_model="models/dataset-4-skincancer/comparaison-cat/", name_model="Mobilnet_Resampling",
	add_top=False, class_mode="categorical", freeze=None, epochs=40, dropout=True,
	classes=7, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, 
	# weight=[0.59633188, 3.09519487],
	weight=[ 4.37527304,  2.78349083,  1.30183284, 12.44099379,  1.28545758, 0.21338021, 10.07545272],
	# load_model="models/dataset-4-skincancer/comparaison/Mobilnet_Resampling/best_model_epoch.hdf5"
)
train.set_train_generator(
	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
)
train.set_val_generator(
	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
)
train.run()




# ---------- Dataset - 4 - skin Cancer -- on Labalised data --------------



# def fn_reader(d, path, size):
# 	# img = Image.open(path + "/" + d["name"]).resize(size)
# 	img = Image.open(path + "/" + d["image_id"]+".jpg").resize(size)
# 	img = light_augmentation(np.array(img))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label_cat"], num_classes=7)]

# def fn_reader_val(d, path, size):
# 	img = np.array(Image.open(path + "/" + d["image_id"]+".jpg").resize(size))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label_cat"], num_classes=7)]


# train = CreateModel(
# 	MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-4-skincancer/comparaison-cat/", name_model="Mobilnet_GAN-labelised",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=60, dropout=True,
# 	classes=7, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, 
# 	# weight=[0.59633188, 3.09519487],
# 	# weight=[ 4.37527304,  2.78349083,  1.30183284, 12.44099379,  1.28545758, 0.21338021, 10.07545272],
# 	load_model="models/dataset-4-skincancer/comparaison-cat/Mobilnet_GAN-labelised/best_model_epoch-02.hdf5"
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# # train.set_train_generator(
# # 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/result_gan/st_labelised_cat__train.csv",
# # 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/result_gan/stylegan"
# # )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()





# ---------- Dataset - 4 - skin Cancer -- pretrain GAN --------------

# def fn_reader(d, path, size):
# 	if d["type"] == "real":
# 		image = np.array(Image.open(path + "/data/all/" + d["image_id"]+".jpg").resize(size)).astype(np.float32)
# 	if d["type"] == "gan":
# 		image =  np.array(Image.open(path + "/result_gan/stylegan/" + d["image_id"]).resize(size)).astype(np.float32)
		
# 	imgs = [ preprocess_input(light_augmentation(image)) ]
# 	labels = [to_categorical(int(d["label_gan"]), num_classes=2)]

# 	return imgs, labels

# def fn_reader_cat(d, path, size):
# 	image = Image.open(path + "/" + d["image_id"]+".jpg")
# 	image = np.array(image.resize(size)).astype(np.float32)
# 	return [preprocess_input(light_augmentation(image))], [to_categorical(int(d["label_cat"]), num_classes=7)]

# def fn_reader_val_cat(d, path, size):
# 	image = Image.open(path + "/" + d["image_id"]+".jpg")
# 	image = np.array(image.resize(size)).astype(np.float32)
# 	return [ preprocess_input(image) ], [ to_categorical(int(d["label_cat"]), num_classes=7) ]


# input_tensor = layers.Input(shape=(224,224,3))
# base_model = MobileNetV2(input_tensor=input_tensor, weights="imagenet", include_top=False)

# x = layers.Conv2D(1280, kernel_size=1, use_bias=False, name='validity_Conv_1')(base_model.get_layer("block_16_project_BN").output)
# x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='validity_Conv_1_bn')(x)
# x = layers.ReLU(6., name='validity_out_relu')(x)
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dropout(0.5)(x)
# validity = layers.Dense(2, activation='softmax')(x)


# x2 = layers.Conv2D(1280,kernel_size=1,use_bias=False,name='classifier_Conv_1')(base_model.get_layer("block_16_project_BN").output)
# x2 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='classifier_Conv_1_bn')(x2)
# x2 = layers.ReLU(6., name='classifier_out_relu')(x2)
# x2 = layers.GlobalAveragePooling2D()(x2)
# x2 = layers.Dropout(0.5)(x2)
# classifier = layers.Dense(7, activation='softmax', name='predictions')(x2)

# ValidityModel = Model(input_tensor, validity)
# ValidityModel.compile(
# 	optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6), 
# 	loss="categorical_crossentropy", 
# 	metrics=['categorical_accuracy']
# )


# ClassifierModel = Model(base_model.input, classifier)
# ClassifierModel.compile(
# 	optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6), 
# 	loss="categorical_crossentropy", 
# 	metrics=['categorical_accuracy']
# )

# for e in range(0,50):
# 	train = CreateModel(
# 		ValidityModel, batch_size=32, batch_size_val=32, save_model=None, name_model="Mobilnet_ACGAN",
# 		class_mode="categorical", freeze=None, epochs=1, custom=True, steps_per_epoch=1000,
# 		classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=None,  weight=[0.60015   , 2.99625562]
# 	)
# 	train.set_train_generator(
# 		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/result_gan/train_real_fake.csv",
# 		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer",
# 	)
# 	train.run()

# 	train = CreateModel(
# 		ClassifierModel, batch_size=32, batch_size_val=32,  save_model="models/dataset-4-skincancer/comparaison-cat/", name_model="Mobilnet_ACGAN_weighted",
# 		class_mode="categorical", freeze=None, epochs=3, custom=True, multi_check=True, ep=e,
# 		classes=7, size=(224,224,3), fn_reader=fn_reader_cat, fn_reader_val=fn_reader_val_cat, 
# 		weight=[ 4.37527304,  2.78349083,  1.30183284, 12.44099379,  1.28545758,0.21338021, 10.07545272],
# 	)
# 	train.set_train_generator(
# 		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# 	)
# 	train.set_val_generator(
# 		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# 	)
# 	train.run()

# 	if e == 25:
# 		ClassifierModel.compile(
# 			optimizer=Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6), 
# 			loss="categorical_crossentropy", 
# 			metrics=['categorical_accuracy']
# 		)
# 		ValidityModel.compile(
# 			optimizer=Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6), 
# 			loss="categorical_crossentropy", 
# 			metrics=['categorical_accuracy']
# 		)


# def fn_reader(d, path, size):
# 	img = Image.open(path + "/" + d["image_id"]+".jpg").resize(size)
# 	img = light_augmentation(np.array(img))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label"], num_classes=2)]

# def fn_reader_val(d, path, size):
# 	img = np.array(Image.open(path + "/" + d["image_id"]+".jpg").resize(size))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label"], num_classes=2)]

# # def fn_reader(d, path, size):
# # 	if d["type"] == "real":
# # 		image = np.array(Image.open(path + "/data/all/" + d["image_id"]+".jpg").resize(size)).astype(np.float32)
# # 	if d["type"] == "gan":
# # 		image =  np.array(Image.open(path + "/result_gan/stylegan/" + d["image_id"]).resize(size)).astype(np.float32)
		
# # 	imgs = [ preprocess_input(light_augmentation(image)) ]
# # 	labels = [to_categorical(int(d["label_gan"]), num_classes=2)]
# # 	return imgs, labels


# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/comparaison/", 
# 	name_model="Mobilnet_Pretrained_weighted",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=40, dropout=True,
# 	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, weight=[0.60015   , 2.99625562],
# 	load_model="models/dataset-4-skincancer/comparaison/Mobilnet_Pretrained_weighted/best_model_epoch_first.hdf5"
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()


# def fn_reader(d, path, size):
# 	img = Image.open(path + "/" + d["image_id"]+".jpg").resize(size)
# 	img = light_augmentation(np.array(img))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label_cat"], num_classes=7)]

# def fn_reader_val(d, path, size):
# 	img = np.array(Image.open(path + "/" + d["image_id"]+".jpg").resize(size))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label_cat"], num_classes=7)]


# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/comparaison-cat/", name_model="Mobilnet_ACGAN_weighted",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=50, dropout=True,
# 	classes=7, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, 
# 	weight=[ 4.37527304,  2.78349083,  1.30183284, 12.44099379,  1.28545758,0.21338021, 10.07545272],
# 	load_model="models/dataset-4-skincancer/comparaison-cat/Mobilnet_ACGAN_weighted/model_epoch-03-9.hdf5"
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()



# ---------- Dataset - 4 - skin Cancer -- Comparaison --------------



# def fn_reader(d, path, size):
# 	img = Image.open(path + "/" + d["image_id"]+".jpg").resize(size)
# 	img = light_augmentation(np.array(img))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label"], num_classes=2)]

# def fn_reader_val(d, path, size):
# 	img = np.array(Image.open(path + "/" + d["image_id"]+".jpg").resize(size))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label"], num_classes=2)]


# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/comparaison/", name_model="mobilnet_224x224_weighted",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=40, dropout=True,
# 	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, weight=[0.62120084, 2.56269191],
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()


# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/comparaison/", name_model="mobilnet_224x224",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=40, dropout=True,
# 	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val,
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()




# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/comparaison/", name_model="mobilnet_96x96_weighted",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=40, dropout=True,
# 	classes=2, size=(96,96,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, weight=[0.62120084, 2.56269191],
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()


# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/comparaison/", name_model="mobilnet_96x96",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=40, dropout=True,
# 	classes=2, size=(96,96,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val,
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()









# def fn_reader_cat(d, path, size):
# 	img = Image.open(path + "/" + d["image_id"]+".jpg").resize(size)
# 	img = light_augmentation(np.array(img))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label_cat"], num_classes=7)]

# def fn_reader_val_cat(d, path, size):
# 	img = np.array(Image.open(path + "/" + d["image_id"]+".jpg").resize(size))
# 	return [preprocess_input(img.astype(np.float32))], [to_categorical(d["label_cat"], num_classes=7)]


# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/comparaison-cat/", name_model="mobilnet_224x224_weighted",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=40, dropout=True,
# 	classes=7, size=(224,224,3), fn_reader=fn_reader_cat, fn_reader_val=fn_reader_val_cat, weight=[ 4.37527304,  2.78349083,  1.30183284, 12.44099379,  1.28545758,
#         0.21338021, 10.07545272],
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()


# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/comparaison-cat/", name_model="mobilnet_224x224",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=40, dropout=True,
# 	classes=7, size=(224,224,3), fn_reader=fn_reader_cat, fn_reader_val=fn_reader_val_cat, weight=[1,1,1,1,1,1,1]
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()


# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/comparaison-cat/", name_model="mobilnet_96x96_weighted",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=40, dropout=True,
# 	classes=7, size=(96,96,3), fn_reader=fn_reader_cat, fn_reader_val=fn_reader_val_cat, weight=[ 4.37527304,  2.78349083,  1.30183284, 12.44099379,  1.28545758,
#         0.21338021, 10.07545272],
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()


# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/comparaison-cat/", name_model="mobilnet_96x96",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=40, dropout=True,
# 	classes=7, size=(96,96,3), fn_reader=fn_reader_cat, fn_reader_val=fn_reader_val_cat, weight=[1,1,1,1,1,1,1]
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()





# os.system('shutdown -s')
# os.system('systemctl poweroff') 



# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/comparaison-Size/", name_model="mobilnet_224x224",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=50, dropout=True,
# 	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val,
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all"
# )
# train.run()






















# def fn_reader(d, path):
# 	# img = Image.open(path + "/" + d["image_id"]+".jpg").resize((224,224))
# 	img = Image.open(path + "/" + d["image_id"]+".jpg").crop((75, 0, 525, 450)).resize((28,28))
# 	img = light_augmentation(np.array(img))
# 	return preprocess_input(img.astype(np.float32)), to_categorical(d["cell_type_idx"], num_classes=7)

# def fn_reader_val(d, path):
# 	# img = np.array(Image.open(path + "/" + d["image_id"]+".jpg").resize((224,224)))
# 	img = np.array(Image.open(path + "/" + d["image_id"]+".jpg").crop((75, 0, 525, 450)).resize((28,28)))
# 	return preprocess_input(img.astype(np.float32)), to_categorical(d["cell_type_idx"], num_classes=7)


# train = CreateModel(
# 	MobileNetV2, batch_size=64, batch_size_val = 64, save_model="models/dataset-4-skincancer/", name_model="MobileNetV2_28x28_weight_600x450_fromscratch",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=150, dropout=True,
# 	classes=7, size=(28,28,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, weight=[4.37527304,  2.78349083,  1.30183284, 12.44099379,  1.28545758,
#         0.21338021, 10.07545272],
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all_600_450"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/val.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all_600_450"
# )
# # train.set_augmentation(get_seq)
# train.run()




# def fn_reader(d, path):
# 	# img = Image.open(path + "/" + d["image_id"]+".jpg").resize((224,224))
# 	img = Image.open(path + "/" + d["image_id"]+".jpg").crop((75, 0, 525, 450)).resize((224,224))
# 	img = light_augmentation(np.array(img))
# 	return preprocess_input(img.astype(np.float32)), to_categorical(d["cell_type_idx"], num_classes=7)

# def fn_reader_val(d, path):
# 	# img = np.array(Image.open(path + "/" + d["image_id"]+".jpg").resize((224,224)))
# 	img = np.array(Image.open(path + "/" + d["image_id"]+".jpg").crop((75, 0, 525, 450)).resize((224,224)))
# 	return preprocess_input(img.astype(np.float32)), to_categorical(d["cell_type_idx"], num_classes=7)


# train = CreateModel(
# 	MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-4-skincancer/", name_model="MobileNetV2_224x224_weight_600x450_2",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=60, dropout=True,
# 	classes=7, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, weight=[4.37527304,  2.78349083,  1.30183284, 12.44099379,  1.28545758,
#         0.21338021, 10.07545272],
# 		load_model="models/dataset-4-skincancer/MobileNetV2_224x224_weight_600x450_2/best_model_epoch_first.hdf5"
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all_600_450"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/val.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-4-skin-cancer/data/all_600_450"
# )
# # train.set_augmentation(get_seq)
# train.run()




# def fn_reader_val(d, path):
# 	img = Image.open(path + "/" + d["patientId"]+".png").resize((224,224))
# 	return np.array(img)*(1./255), to_categorical(d["Target"], num_classes=2)

# def fn_reader(d, path):
# 	img = Image.open(path + "/" + d["patientId"]+".png").resize((224,224))
# 	img = light_augmentation(np.array(img))
# 	return img*(1./255), to_categorical(d["Target"], num_classes=2)

# train = CreateModel(
# 	MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-2/", name_model="MobileNetV2_224x224",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=25, dropout=True,
# 	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/data/all_rgb"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/data/all_rgb"
# )
# # train.set_augmentation(get_seq)
# train.run()



# train = CreateModel(
# 	DenseNet121, batch_size=12, batch_size_val = 12, save_model="models/dataset-2/", name_model="DenseNet121_224x224",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=25, dropout=True,
# 	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/data/all_rgb"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/data/all_rgb"
# )
# # train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
# 	ResNet50, batch_size=12, batch_size_val = 12, save_model="models/dataset-2/", name_model="ResNet50_224x224",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=25, dropout=True,
# 	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/data/all_rgb"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/data/all_rgb"
# )
# # train.set_augmentation(get_seq)
# train.run()



# train = CreateModel(
# 	InceptionResNetV2, batch_size=12, batch_size_val = 12, save_model="models/dataset-2/", name_model="InceptionResNetV2_224x224",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=25, dropout=True,
# 	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/data/all_rgb"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/data/all_rgb"
# )
# # train.set_augmentation(get_seq)
# train.run()



# train = CreateModel(
# 	NASNetMobile, batch_size=12, batch_size_val = 12, save_model="models/dataset-2/", name_model="NASNetMobile_224x224",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=25, dropout=True,
# 	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/data/all_rgb"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-2/data/all_rgb"
# )
# # train.set_augmentation(get_seq)
# train.run()