from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Activation, concatenate, AveragePooling2D, Lambda, MaxPooling2D, Conv2D, BatchNormalization, GlobalAveragePooling2D, ReLU, AveragePooling2D, Permute, Activation
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa
from keras.optimizers import Adam, SGD
from keras.layers import Input
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import h5py
from sklearn.utils import class_weight
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from input_sequence import SequenceFromPandas
from PIL import Image



class CreateModel:
	
	def __init__(self, 
				 CusModel, dropout=False, size=(224, 224, 3), save_model="models/", name_model = "01", class_mode="binary",
				 lr = 1e-4, batch_size=40, batch_size_val = 20, epochs=100, weight = [1,1], add_top=True, imagenet=None, freeze=None, 
				 multi_check=False, dense_neurons=1024, dense_layer=1, classes=2, y_col="label", load_model=None, fn_reader=None, fn_reader_val=None,
				 custom=False, steps_per_epoch=None, ep=0):
		self.save_model = save_model
		self.lr = lr
		self.batch_size = batch_size
		self.batch_size_val = batch_size_val
		self.epochs = epochs
		self.Model = CusModel
		self.size=size
#         self.weight = [1.7293698, 0.70335622]
		self.weight = weight
		self.imagenet = imagenet
		self.freeze = freeze
		self.steps_per_epoch = steps_per_epoch
		self.ep = ep
		
		self.dense_neurons = dense_neurons
		self.dense_layer = dense_layer
		
		self.train_generator = None
		self.val_generator = None
		
		self.size_train = 1
		self.size_val = 1
		
		self.name_model = name_model
		self.dropout = dropout
		self.class_mode = class_mode
		self.add_top = add_top
		self.augmentation = None
		self.multi_check = multi_check
		self.load_model = load_model
		self.custom = custom

		self.classes = classes
		self.y_col = y_col
		self.fn_reader = fn_reader if fn_reader else self.fn_reader_default
		self.fn_reader_val = fn_reader_val

	def set_augmentation(self, augmentation):
		self.augmentation = augmentation

	def fn_reader_default(self, d, path):
		img = np.array(Image.open(path + "/" + d["name"])) * (1./255)
		return img, to_categorical(d["label"], num_classes=2)

	def set_train_generator(self, path_csv, path_train):
		df = pd.read_csv(path_csv)
		sequence = SequenceFromPandas(
			df,
			self.fn_reader,
			batch_size=self.batch_size,
			shuffle=True,
			kwargs={
				"path" : path_train,
				"size" : self.size[:2]
			}
		)
		self.train_generator = sequence
		self.size_train = len(df)
	
	def set_val_generator(self, path_csv, path_val):
		df = pd.read_csv(path_csv)
		sequence = SequenceFromPandas(
			df,
			self.fn_reader_val,
			batch_size=self.batch_size,
			shuffle=True,
			kwargs={
				"path": path_val,
				"size" : self.size[:2]
			}
		)
		self.val_generator = sequence
		self.size_val = len(df)
	
	def run(self):
		if self.custom:
			model = self.Model
		else:
			input_tensor = Input(shape=self.size)
			base_model = self.Model(input_tensor=input_tensor, weights="imagenet", include_top=False)
	
			if self.add_top == "fc":
				x = GlobalAveragePooling2D()(base_model.output)
				x = Dropout(0.5)(x)
				x = Dense(1024, activation='relu')(x)
				x = Dropout(0.5)(x)
				x = Dense(self.classes, activation='softmax', name='predictions')(x)
			else:
				x = GlobalAveragePooling2D()(base_model.output)
				x = Dropout(0.5)(x)
				x = Dense(self.classes, activation='softmax', name='predictions')(x)
			
			model = Model(base_model.input, x)
		
		if self.load_model:
			model.load_weights(self.load_model)
		
		if self.freeze:
			if self.freeze == "auto":
				last_layer = len(base_model.layers)
				print(" ------------------ : ", last_layer)
				for layer in model.layers[:last_layer]:
					layer.trainable = False
				for layer in model.layers[last_layer:]:
					layer.trainable = True
			else:
				for layer in model.layers[:self.freeze]:
					layer.trainable = False
				for layer in model.layers[self.freeze:]:
					layer.trainable = True
		
		# model.summary()
 
		if self.class_mode == "binary":
			lss = 'binary_crossentropy'
			mtr = 'val_binary_accuracy'
		if self.class_mode == "categorical":
			lss = 'categorical_crossentropy'
			mtr = 'val_categorical_accuracy'
				
		model.compile(
			optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6), 
			loss=lss, 
			metrics=['categorical_accuracy']
		)
		
		if self.save_model:
			if not os.path.exists(self.save_model+self.name_model):
				os.makedirs(self.save_model+self.name_model)
			
			if self.multi_check:
				checkpoint = ModelCheckpoint(
					self.save_model+self.name_model+"/model_epoch-{epoch:02d}-"+str(self.ep)+".hdf5", monitor=mtr, verbose=1, save_best_only=False, mode='max')
			else:
				checkpoint = ModelCheckpoint(
					self.save_model+self.name_model+"/best_model_epoch.hdf5", monitor=mtr, verbose=1, save_best_only=True, mode='max')
				
			earlystop = EarlyStopping(monitor='val_loss',min_delta=0,patience=5,verbose=0, mode='auto')
			
			tbCallBack = TensorBoard(
				log_dir='tensorboard/'+self.save_model.replace("models","")+self.name_model,
				histogram_freq=0, write_graph=True, write_images=True
			)

			callbacks_list = [tbCallBack, checkpoint]
		else:
			callbacks_list = []
		
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                              patience=5, min_lr=0.00001, verbose=1)
		callbacks_list.append(reduce_lr)

		
		model.fit_generator(
			self.train_generator,
			epochs=self.epochs,
			validation_data=self.val_generator,
            class_weight=self.weight,
			callbacks=callbacks_list,
			steps_per_epoch=self.steps_per_epoch,
			# steps_per_epoch=int(self.size_train/self.batch_size),
			# validation_steps=int(self.size_val/self.batch_size_val),
			use_multiprocessing=True,
			max_queue_size=60,
			workers=30
		)
		K.clear_session()