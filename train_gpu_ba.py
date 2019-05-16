import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3 
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from imgaug import augmenters as iaa
import random
from keras import backend as K
from create_model_dir import CreateModel

def get_seq():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                shear=(-5, 5), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                    ]),
                    iaa.Invert(0.01, per_channel=True), # invert color channels
                    iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.9, 1.1), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-1, 0),
                            first=iaa.Multiply((0.9, 1.1), per_channel=True),
                            second=iaa.ContrastNormalization((0.9, 1.1))
                        )
                    ]),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq





################################# 224x224 ##################################
############################################################################

# ---------- Sans Aug ------------------ 


# train = CreateModel(
#     NASNetMobile(input_shape=(224,224,3), classes=2, include_top=False), batch_size=32, batch_size_val = 32,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="NASNetMobile_224x224_noaug", multi_check=False, epochs=10,
# 	class_mode="categorical" )
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()

# train = CreateModel(
#     NASNetLarge(input_shape=(224,224,3), classes=2, include_top=False), batch_size=12, batch_size_val = 12,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="NASNetLarge_224x224_noaug", multi_check=False, epochs=10,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
#     InceptionResNetV2(input_shape=(224,224,3), classes=2, include_top=False), batch_size=12, batch_size_val = 12,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="InceptionResNetV2_224x224_noaug", multi_check=False, epochs=10,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()

# train = CreateModel(
#     DenseNet121(input_shape=(224,224,3), classes=2, include_top=False), batch_size=12, batch_size_val = 12,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="DenseNet121_224x224_noaug", multi_check=False, epochs=10,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
#     ResNet50(input_shape=(224,224,3), classes=2, include_top=False), batch_size=12, batch_size_val = 12,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="ResNet50_224x224_noaug", multi_check=False, epochs=10,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


# ---------- Avec Aug  2 ------------------ 


# train = CreateModel(
#     NASNetMobile(input_shape=(224,224,3), classes=2, include_top=False), batch_size=32, batch_size_val = 32,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="NASNetMobile_224x224_lightaug", multi_check=False, epochs=15,
# 	class_mode="categorical" )
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
#     NASNetLarge(input_shape=(224,224,3), classes=2, include_top=False), batch_size=18, batch_size_val = 18,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="NASNetLarge_224x224_lightaug", multi_check=False, epochs=15,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
#     InceptionResNetV2(input_shape=(224,224,3), classes=2, include_top=False), batch_size=12, batch_size_val = 12,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="InceptionResNetV2_224x224_lightaug", multi_check=False, epochs=15,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()

# train = CreateModel(
#     DenseNet121(input_shape=(224,224,3), classes=2, include_top=False), batch_size=12, batch_size_val = 12,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="DenseNet121_224x224_lightaug", multi_check=False, epochs=15,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
#     ResNet50(input_shape=(224,224,3), classes=2, include_top=False), batch_size=12, batch_size_val = 12,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="ResNet50_224x224_lightaug", multi_check=False, epochs=15,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


# ---------- Avec Aug ------------------ 


# train = CreateModel(
#     NASNetMobile(input_shape=(224,224,3), classes=2, include_top=False), batch_size=32, batch_size_val = 32,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="NASNetMobile_224x224", multi_check=False, epochs=20,
# 	class_mode="categorical" )
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# train.set_augmentation(get_seq)
# train.run()

# train = CreateModel(
#     NASNetLarge(input_shape=(224,224,3), classes=2, include_top=False), batch_size=32, batch_size_val = 32,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="NASNetLarge_224x224", multi_check=False, epochs=20,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
#     InceptionResNetV2(input_shape=(224,224,3), classes=2, include_top=False), batch_size=24, batch_size_val = 24,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="InceptionResNetV2_224x224", multi_check=False, epochs=20,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# train.set_augmentation(get_seq)
# train.run()

# train = CreateModel(
#     DenseNet121(input_shape=(224,224,3), classes=2, include_top=False), batch_size=24, batch_size_val = 24,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="DenseNet121_224x224", multi_check=False, epochs=20,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
#     ResNet50(input_shape=(224,224,3), classes=2, include_top=False), batch_size=24, batch_size_val = 24,  size=(224,224,3),
#     save_model="models/dataset-1/", name_model="ResNet50_224x224", multi_check=False, epochs=20,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# train.set_augmentation(get_seq)
# train.run()










################################# 96x96 ####################################
############################################################################


# ---------- Avec Aug ------------------ 



# train = CreateModel(
#     NASNetMobile(input_shape=(96,96,3), classes=2, include_top=False), batch_size=24, batch_size_val = 24,  size=(96,96,3),
#     save_model="models/dataset-1/", name_model="NASNetMobile_96x96", multi_check=False, epochs=15,
# 	class_mode="categorical" )
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()

# train = CreateModel(
#     NASNetLarge(input_shape=(96,96,3), classes=2, include_top=False), batch_size=18, batch_size_val = 18,  size=(96,96,3),
#     save_model="models/dataset-1/", name_model="NASNetLarge_96x96", multi_check=False, epochs=15,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
#     InceptionResNetV2(input_shape=(96,96,3), classes=2, include_top=False), batch_size=18, batch_size_val = 18,  size=(96,96,3),
#     save_model="models/dataset-1/", name_model="InceptionResNetV2_96x96", multi_check=False, epochs=15,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()

# train = CreateModel(
#     DenseNet121(input_shape=(96,96,3), classes=2, include_top=False), batch_size=18, batch_size_val = 18,  size=(96,96,3),
#     save_model="models/dataset-1/", name_model="DenseNet121_96x96", multi_check=False, epochs=15,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
#     ResNet50(input_shape=(96,96,3), classes=2, include_top=False), batch_size=18, batch_size_val = 18,  size=(96,96,3),
#     save_model="models/dataset-1/", name_model="ResNet50_96x96", multi_check=False, epochs=15,
# 	class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()





# ---------- Sans Aug ------------------ 

#train = CreateModel(
#     NASNetMobile(input_shape=(96,96,3), classes=2, include_top=False), batch_size=24, batch_size_val = 24,  size=(96,96,3),
#     save_model="models/dataset-1/", name_model="NASNetMobile_96x96_noaug", multi_check=False, epochs=15,
#     class_mode="categorical" )
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()

# train = CreateModel(
#     NASNetLarge(input_shape=(96,96,3), classes=2, include_top=False), batch_size=18, batch_size_val = 18,  size=(96,96,3),
#     save_model="models/dataset-1/", name_model="NASNetLarge_96x96_noaug", multi_check=False, epochs=15,
#     class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
#     InceptionResNetV2(input_shape=(96,96,3), classes=2, include_top=False), batch_size=18, batch_size_val = 18,  size=(96,96,3),
#     save_model="models/dataset-1/", name_model="InceptionResNetV2_96x96_noaug", multi_check=False, epochs=15,
#     class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()

# train = CreateModel(
#     DenseNet121(input_shape=(96,96,3), classes=2, include_top=False), batch_size=18, batch_size_val = 18,  size=(96,96,3),
#     save_model="models/dataset-1/", name_model="DenseNet121_96x96_noaug", multi_check=False, epochs=15,
#     class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


# train = CreateModel(
#     ResNet50(input_shape=(96,96,3), classes=2, include_top=False), batch_size=18, batch_size_val = 18,  size=(96,96,3),
#     save_model="models/dataset-1/", name_model="ResNet50_96x96_noaug", multi_check=False, epochs=15,
#     class_mode="categorical")
# train.set_train_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/train")
# train.set_val_generator("/home/yannis/Developpement/ppd-GAN-for-medical-imaging/dataset-1/data/val")
# # train.set_augmentation(get_seq)
# train.run()


