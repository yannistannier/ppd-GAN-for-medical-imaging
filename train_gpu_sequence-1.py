import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3 
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
#from keras.applications.densenet import DenseNet121, preprocess_input as preprocess_input_densent
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from imgaug import augmenters as iaa
from keras.utils.np_utils import to_categorical
import random
import numpy as np
import pydicom
from keras import backend as K
from keras import layers, Model
from keras.optimizers import Adam

from create_model import CreateModel
# from create_model_cust import CreateModel as CreateModelCust

from PIL import Image


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



# ------------ Training 7 classes + multi output ------------------
def fn_reader(d, path):
	size = (224,224)
	if d["type"] == "real":
		image = np.array(Image.open(path + "/data/original/p3/" + d["name"]).resize(size)).astype(np.float32)
	if d["type"] == "pggan" or d["type"] == "stylegan":
		image =  np.array(Image.open(path + "/result_gan/"+str(d["type"])+"/" + d["name"]).resize(size)).astype(np.float32)
		
	imgs = [ preprocess_input(light_augmentation(image)) ]
	labels = [to_categorical(int(d["label"]), num_classes=2)]

	return imgs, labels

def fn_reader_cat(d, path):
	size = (224,224)
	image = Image.open(path + "/" + d["name"])
	image = np.array(image.resize(size)).astype(np.float32)
	return [preprocess_input(light_augmentation(image))], [to_categorical(int(d["label_cat"]), num_classes=8)]

def fn_reader_val_cat(d, path):
	size = (224,224)
	image = Image.open(path + "/" + d["name"])
	image = np.array(image.resize(size)).astype(np.float32)
	return [ preprocess_input(image) ], [ to_categorical(int(d["label_cat"]), num_classes=8) ]


input_tensor = layers.Input(shape=(224,224,3))
base_model = MobileNetV2(input_tensor=input_tensor, weights="imagenet", include_top=False)

x = layers.Conv2D(1280, kernel_size=1, use_bias=False, name='validity_Conv_1')(base_model.get_layer("block_16_project_BN").output)
x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='validity_Conv_1_bn')(x)
x = layers.ReLU(6., name='validity_out_relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
validity = layers.Dense(2, activation='softmax')(x)


x2 = layers.Conv2D(1280,kernel_size=1,use_bias=False,name='classifier_Conv_1')(base_model.get_layer("block_16_project_BN").output)
x2 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='classifier_Conv_1_bn')(x2)
x2 = layers.ReLU(6., name='classifier_out_relu')(x2)
x2 = layers.GlobalAveragePooling2D()(x2)
x2 = layers.Dropout(0.5)(x2)
classifier = layers.Dense(8, activation='softmax', name='predictions')(x2)

ValidityModel = Model(input_tensor, validity)
ValidityModel.compile(
	optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6), 
	loss="categorical_crossentropy", 
	metrics=['categorical_accuracy']
)


ClassifierModel = Model(base_model.input, classifier)
# ClassifierModel.load_weights("models/dataset-1/Comparaison-Cat/Mobilnet_ACGAN/model_epoch-16.hdf5")
ClassifierModel.compile(
	optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6), 
	loss="categorical_crossentropy", 
	metrics=['categorical_accuracy']
)






for e in range(0,50):
	train = CreateModel(
		ValidityModel, batch_size=32, batch_size_val=32, save_model=None, name_model="Mobilnet_ACGAN",
		class_mode="categorical", freeze=None, epochs=1, custom=True, steps_per_epoch=1200,
		classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=None
	)
	train.set_train_generator(
		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/acgan/train_real_fake.csv",
		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1",
	)
	train.run()

	train = CreateModel(
		ClassifierModel, batch_size=32, batch_size_val=32,  save_model=None, name_model="Mobilnet_ACGAN",
		class_mode="categorical", freeze=None, epochs=1, custom=True, multi_check=True, ep=e,
		classes=8, size=(224,224,3), fn_reader=fn_reader_cat, fn_reader_val=fn_reader_val_cat, 
		weight=[0.26589304, 0.99309045, 1.30877483, 1.80892449, 1.68191489, 1.57157058, 3.62614679, 2.44736842]
	)
	train.set_train_generator(
		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/train_p3.csv",
		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/p3",
	)
	train.set_val_generator(
		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test_p3.csv",
		"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/p3",
	)
	train.run()




exit()
# ------------ Training 7 classes ------------------

# def fn_reader(d, path):
# 	size = (224,224)
# 	image = Image.open(path + "/" + d["name"])
# 	# img1 = np.array(image.crop((20, 0, 460+20, 460)).resize(size)).astype(np.float32)
# 	img2 = np.array(image.crop((120, 0, 580, 460)).resize(size)).astype(np.float32)
# 	# img3 = np.array(image.crop((240-20, 0, 700-20, 460)).resize(size)).astype(np.float32)

# 	imgs = [ preprocess_input(light_augmentation(img)) for img in [img2] ]
# 	labels = [to_categorical(int(d["label_cat"]), num_classes=8) for x in range(0,len(imgs))]

# 	return imgs, labels

# def fn_reader_val(d, path):
# 	size = (224,224)
# 	image = Image.open(path + "/" + d["name"])
# 	image = np.array(image.crop((120, 0, 580, 460)).resize(size)).astype(np.float32)
# 	return [preprocess_input(image)], [to_categorical(int(d["label_cat"]), num_classes=8)]


# train = CreateModel(
#   MobileNetV2, batch_size=30, batch_size_val = 30, save_model="models/dataset-1/Comparaison-Cat/", name_model="Mobilnet_p1",
#   add_top=False, class_mode="categorical", freeze=None, epochs=25, dropout=True,
#   classes=8, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, weight=[0.26589304, 0.99309045, 1.30877483, 1.80892449, 1.68191489,
#        1.57157058, 3.62614679, 2.44736842]
# )
# train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/train.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# train.run()



# ------------ Training Comparaison size ------------------

# def fn_reader(d, path):
#     size = (224,224)
#     if d["type"] == "pggan" or d["type"] == "stylegan":
#         image = Image.open(path + "/result_gan/"+str(d["type"])+"/" + d["name"]).resize(size)
#         image = np.array(image).astype(np.float32)
#         imgs = [ preprocess_input(light_augmentation(image)) ]
#         labels = [ to_categorical(int(d["label"]), num_classes=2) ]
#     return imgs, labels

# def fn_reader_val(d, path):
#     size = (224,224)
#     image = Image.open(path + "/" + d["name"])
#     img1 = np.array(image.crop((20, 0, 460+20, 460)).resize(size)).astype(np.float32)
#     img2 = np.array(image.crop((120, 0, 580, 460)).resize(size)).astype(np.float32)
#     img3 = np.array(image.crop((240-20, 0, 700-20, 460)).resize(size)).astype(np.float32)

#     imgs = [ preprocess_input(img) for img in [img1, img2, img3] ]
#     labels = [to_categorical(int(d["label"]), num_classes=2) for x in range(0,3)]
#     return imgs, labels


# size=["7k", "12k", "20k", "30k"]
# tp = ["train_mixgan"] # "train_pgan", "train_mixgan"
# # done : train_stylegan, train_pgan

# for t in tp:
#     for s in size:
#         csv = t+"-"+s+".csv"
#         name = t.split('_')[1]+"-"+s
        
#         train = CreateModel(
#             MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-1/Comparaison-Size/", 
#             name_model=name, add_top=False, class_mode="categorical", freeze=None, 
#             epochs=25, dropout=True, classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
#         )
#         train.set_train_generator(
#            "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/"+csv,
#            "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1"
#         )
#         train.set_val_generator(
#            "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#            "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
#         )
#         # # train.set_augmentation(get_seq)
#         train.run()


# os.system('shutdown -s')
# os.system('systemctl poweroff') 
# exit()

# # ------------ Training on Gan Image + Real ----------------

# def fn_reader(d, d2, path):
# 	size = (224,224)
	
# 	image = Image.open(path + "/data/original/all/" + d["name"])
# 	img1 = np.array(image.crop((20, 0, 460+20, 460)).resize(size)).astype(np.float32)
# 	img2 = np.array(image.crop((120, 0, 580, 460)).resize(size)).astype(np.float32)
# 	img3 = np.array(image.crop((240-20, 0, 700-20, 460)).resize(size)).astype(np.float32)
	
# 	imgs = [ preprocess_input(light_augmentation(img1)),
# 			 preprocess_input(light_augmentation(img2)),
# 			 preprocess_input(light_augmentation(img3))
# 	]
# 	labels = [to_categorical(int(d["label"]), num_classes=2), to_categorical(int(d["label"]), num_classes=2), to_categorical(int(d["label"]), num_classes=2)]
	
# 	for x in d2:
# 		image = Image.open(path + "/result_gan/"+str(x["type"])+"/" + x["name"]).resize(size)
# 		image = np.array(image).astype(np.float32)
# 		imgs.append( preprocess_input(light_augmentation(image)) )
# 		labels.append( to_categorical(int(x["label"]), num_classes=2) )
	
# 	return imgs, labels

# def fn_reader_val(d, path):
# 	size = (224,224)
# 	image = Image.open(path + "/" + d["name"])
# 	img1 = np.array(image.crop((20, 0, 460+20, 460)).resize(size)).astype(np.float32)
# 	img2 = np.array(image.crop((120, 0, 580, 460)).resize(size)).astype(np.float32)
# 	img3 = np.array(image.crop((240-20, 0, 700-20, 460)).resize(size)).astype(np.float32)

# 	imgs = [ preprocess_input(img) for img in [img1, img2, img3] ]
# 	labels = [to_categorical(int(d["label"]), num_classes=2) for x in range(0,3)]
# 	return imgs, labels

# train = CreateModelCust(
# 	MobileNetV2, batch_size=5, batch_size_val = 5, save_model="models/dataset-1/Comparaison_2/", name_model="Mobilnet_real-mixgan_n8_3",
# 	add_top=False, class_mode="categorical", freeze=None, epochs=25, dropout=True,
#     load_model="models/dataset-1/Comparaison_2/Mobilnet_real-mixgan_n8_3/best_model_epoch-first.hdf5",
# 	classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, add_nb=8
# )
# train.set_train_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/train.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/train_mixgan.csv"
# )
# train.set_val_generator(
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
# 	"/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# # # train.set_augmentation(get_seq)
# train.run()





# train = CreateModelCust(
#   MobileNetV2, batch_size=6, batch_size_val = 6, save_model="models/dataset-1/Comparaison_2/", name_model="Mobilnet_real-pggan_n6",
#   add_top=False, class_mode="categorical", freeze=None, epochs=25, dropout=True,
#   classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, add_nb=6
# )
# train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/train.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/train_pgan.csv"
# )
# train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# # train.set_augmentation(get_seq)
# train.run()



#train = CreateModelCust(
#   MobileNetV2, batch_size=6, batch_size_val = 6, save_model="models/dataset-1/Comparaison_2/", name_model="Mobilnet_real-stylegan_n6",
#   add_top=False, class_mode="categorical", freeze=None, epochs=25, dropout=True,
#   classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val, add_nb=6
#)
#train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/train.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/train_stylegan-rdm.csv"
#)
#train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
#)
# train.set_augmentation(get_seq)
#train.run()







# def fn_reader(d, path):
#   size = (224,224)
	
#   if d["type"] == "real":
#       image = Image.open(path + "/data/original/all/" + d["name"])
#       img1 = np.array(image.crop((20, 0, 460+20, 460)).resize(size)).astype(np.float32)
#       img2 = np.array(image.crop((120, 0, 580, 460)).resize(size)).astype(np.float32)
#       img3 = np.array(image.crop((240-20, 0, 700-20, 460)).resize(size)).astype(np.float32)

#       imgs = [ preprocess_input(light_augmentation(img)) for img in [img1, img2, img3] ]
#       labels = [to_categorical(int(d["label"]), num_classes=2) for x in range(0,3)]
		
#   if d["type"] == "pggan" or d["type"] == "stylegan":
#       image = Image.open(path + "/result_gan/"+str(d["type"])+"/" + d["name"]).resize(size)
#       image = np.array(image).astype(np.float32)
#       imgs = [ preprocess_input(light_augmentation(image)) ]
#       labels = [ to_categorical(int(d["label"]), num_classes=2) ]
	
#   return imgs, labels

# def fn_reader_val(d, path):
#   size = (224,224)
#   image = Image.open(path + "/" + d["name"])
#   img1 = np.array(image.crop((20, 0, 460+20, 460)).resize(size)).astype(np.float32)
#   img2 = np.array(image.crop((120, 0, 580, 460)).resize(size)).astype(np.float32)
#   img3 = np.array(image.crop((240-20, 0, 700-20, 460)).resize(size)).astype(np.float32)

#   imgs = [ preprocess_input(img) for img in [img1, img2, img3] ]
#   labels = [to_categorical(int(d["label"]), num_classes=2) for x in range(0,3)]
#   return imgs, labels



# train = CreateModel(
#   MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-1/Comparaison/", name_model="Mobilnet_real-stylegan20K_224x224",
#   add_top=False, class_mode="categorical", freeze=None, epochs=35, dropout=True,
#   classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/train_real-stylegan20k.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1"
# )
# train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# # train.set_augmentation(get_seq)
# train.run()



# train = CreateModelCust(
#   MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-1/Comparaison/", name_model="Mobilnet_real-pggan_224x224",
#   add_top=False, class_mode="categorical", freeze=None, epochs=35, dropout=True,
#   classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/train_real-pggan.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1"
# )
# train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# # train.set_augmentation(get_seq)
# train.run()




exit()

# train = CreateModel(
#   MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-1/Comparaison/", name_model="Mobilnet_real-mixgan_224x224",
#   add_top=False, class_mode="categorical", freeze=None, epochs=50, dropout=True,
#   classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/train_real_mixgan.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1"
# )
# train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# # train.set_augmentation(get_seq)
# train.run()



# --------- Training on Gan Image ----------------


#def fn_reader(d, path):
#   size = (224,224)
#   image = np.array(Image.open(path + "/" + d["type"] + "/" + d["name"]).resize(size)).astype(np.float32)
#   return preprocess_input(light_augmentation(image)), to_categorical(int(d["label"]), num_classes=2)

#def fn_reader_val(d, path):
#   size = (224,224)
#   image = Image.open(path + "/" + d["name"])
#   image = np.array(image.crop((120, 0, 580, 460)).resize(size)).astype(np.float32)
#   return preprocess_input(image), to_categorical(int(d["label"]), num_classes=2)


#train = CreateModel(
#   MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-1/Comparaison/", name_model="Mobilnet_only-mixgan_224x224",
#   add_top=False, class_mode="categorical", freeze=None, epochs=25, dropout=True,
#   classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
#)
#train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/train_mixgan.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan"
#)
#train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
#)
# train.set_augmentation(get_seq)
#train.run()


# def fn_reader(d, path):
#   size = (224,224)
#   image = np.array(Image.open(path + "/" + d["name"]).resize(size)).astype(np.float32)
#   return preprocess_input(light_augmentation(image)), to_categorical(int(d["label"]), num_classes=2)

# def fn_reader_val(d, path):
#   size = (224,224)
#   image = Image.open(path + "/" + d["name"])
#   image = np.array(image.crop((120, 0, 580, 460)).resize(size)).astype(np.float32)
#   return preprocess_input(image), to_categorical(int(d["label"]), num_classes=2)
	

# train = CreateModel(
#   MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-1/Comparaison/", name_model="Mobilnet_only-pgan_224x224",
#   add_top=False, class_mode="categorical", freeze=None, epochs=50, dropout=True,
#   classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/train_pgan.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/pggan"
# )
# train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# # train.set_augmentation(get_seq)
# train.run()



# train = CreateModel(
#   MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-1/Comparaison/", name_model="Mobilnet_only-stylegan-rdm_224x224",
#   add_top=False, class_mode="categorical", freeze=None, epochs=30, dropout=True,
#   classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/train_stylegan.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/result_gan/stylegan"
# )
# train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# # train.set_augmentation(get_seq)
# train.run()


exit()



# ---------- Dataset - Breast Cancer --------------

# def fn_reader(d, path):
#   size = (224,224)
#   image = np.array(Image.open(path + "/" + d["name"]).resize(size)).astype(np.float32)
#   return preprocess_input(light_augmentation(image)), to_categorical(int(d["label"]), num_classes=2)

# def fn_reader_val(d, path):
#   size = (224,224)
#   image = np.array(Image.open(path + "/" + d["name"]).resize(size)).astype(np.float32)
#   return preprocess_input(image), to_categorical(int(d["label"]), num_classes=2)


# train = CreateModel(
#   MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-1/Comparaison/", name_model="Mobilnet_p0_original_224x224",
#   add_top=False, class_mode="categorical", freeze=None, epochs=30, dropout=True,
#   classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/train.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# # train.set_augmentation(get_seq)
# train.run()


# def fn_reader2(d, path):
#   size = (128,128)
#   image = np.array(Image.open(path + "/" + d["name"]).resize(size)).astype(np.float32)
#   return preprocess_input(light_augmentation(image)), to_categorical(int(d["label"]), num_classes=2)

# def fn_reader_val2(d, path):
#   size = (128,128)
#   image = np.array(Image.open(path + "/" + d["name"]).resize(size)).astype(np.float32)
#   return preprocess_input(image), to_categorical(int(d["label"]), num_classes=2)

# train = CreateModel(
#   MobileNetV2, batch_size=32, batch_size_val = 32, save_model="models/dataset-1/Comparaison/", name_model="Mobilnet_p0_original_128x128",
#   add_top=False, class_mode="categorical", freeze=None, epochs=30, dropout=True,
#   classes=2, size=(128,128,3), fn_reader=fn_reader2, fn_reader_val=fn_reader_val2
# )
# train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/train.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/original/all"
# )
# # train.set_augmentation(get_seq)
# train.run()







# def fn_reader(d, path):
#   size = (224,224)
#   image = Image.open(path + "/" + d["name"])
#   img1 = np.array(image.crop((20, 0, 460+20, 460)).resize(size)).astype(np.float32)
#   img2 = np.array(image.crop((120, 0, 580, 460)).resize(size)).astype(np.float32)
#   img3 = np.array(image.crop((240-20, 0, 700-20, 460)).resize(size)).astype(np.float32)

#   imgs = [ preprocess_input_densent(light_augmentation(img)) for img in [img1, img2, img3] ]
#   labels = [to_categorical(int(d["label"]), num_classes=2) for x in range(0,3)]
#   return imgs, labels

# def fn_reader_val(d, path):
#   size = (224,224)
#   image = Image.open(path + "/" + d["name"])
#   img1 = np.array(image.crop((20, 0, 460+20, 460)).resize(size)).astype(np.float32)
#   img2 = np.array(image.crop((120, 0, 580, 460)).resize(size)).astype(np.float32)
#   img3 = np.array(image.crop((240-20, 0, 700-20, 460)).resize(size)).astype(np.float32)

#   imgs = [ preprocess_input_densent(img) for img in [img1, img2, img3] ]
#   labels = [to_categorical(int(d["label"]), num_classes=2) for x in range(0,3)]
#   return imgs, labels



# train = CreateModel(
#   DenseNet121, batch_size=8, batch_size_val = 8, save_model="models/dataset-1/Comparaison/", name_model="DenseNet121_p3_original_224x224",
#   add_top=False, class_mode="categorical", freeze=None, epochs=30, dropout=True,
#   classes=2, size=(224,224,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/train.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/normalize/all"
# )
# train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/normalize/all"
# )
# # # train.set_augmentation(get_seq)
# train.run()



# train = CreateModel(
#   MobileNetV2, batch_size=8, batch_size_val = 8, save_model="models/dataset-1/Comparaison/", name_model="Mobilnet_p3_norm_128x128",
#   add_top=False, class_mode="categorical", freeze=None, epochs=30, dropout=True,
#   classes=2, size=(128,128,3), fn_reader=fn_reader, fn_reader_val=fn_reader_val
# )
# train.set_train_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/train.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/normalize/all"
# )
# train.set_val_generator(
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/test.csv",
#   "/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/normalize/all"
# )
# # train.set_augmentation(get_seq)
# train.run()



