from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Activation, concatenate, AveragePooling2D, Lambda, MaxPooling2D, Conv2D, BatchNormalization, GlobalAveragePooling2D, ReLU
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa
from keras.optimizers import Adam, SGD
from keras.layers import Input
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
import h5py
from sklearn.utils import class_weight
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import random
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class CreateModel:
	
	def __init__(self, 
				 CusModel, dropout=False, size=(224, 224, 3), save_model="models/", name_model = "01", class_mode="binary",
				 lr = 1e-4, batch_size=40, batch_size_val = 20, epochs=100, weight = [1,1], add_top=False, imagenet=None, freeze=None, 
				 multi_check=True, dense_neurons=1024, dense_layer=1):
		self.save_model = save_model
		self.lr = lr
		self.batch_size = batch_size
		self.batch_size_val = batch_size_val
		self.epochs = epochs
		self.Model = CusModel
		self.size=size
		self.weight = weight
		self.imagenet = imagenet
		self.freeze = freeze
		
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

	def set_augmentation(self, augmentation):
		self.augmentation = augmentation
	
	def set_train_generator(self, path_train):
		train_datagen = ImageDataGenerator(
			rescale=(1/255),
			rotation_range=10,
			width_shift_range=0.1,
			height_shift_range=0.1,
			shear_range=0.01,
			zoom_range=[0.8, 1.2],
			horizontal_flip=True,
			vertical_flip=True,
			fill_mode='nearest',
			brightness_range=[0.9, 1.2],
			# preprocessing_function=self.augmentation
		)
		self.train_generator = train_datagen.flow_from_directory(
			path_train,
			target_size=self.size[:-1],
			batch_size=self.batch_size,
			class_mode=self.class_mode
		)
		self.size_train = len(self.train_generator.filenames)
	
	def set_val_generator(self, path_val):
		val_datagen = ImageDataGenerator(rescale=(1/255))
		self.val_generator = val_datagen.flow_from_directory(
			path_val,
			target_size=self.size[:-1],
			batch_size=self.batch_size,
			class_mode=self.class_mode
		)
		self.size_val = len(self.val_generator.filenames) 

	def run(self):
		base_model = self.Model
		if self.add_top == "fc":
			x = GlobalAveragePooling2D()(base_model.output)
			x = Dropout(0.5)(x)
			x = Dense(1024, activation='relu')(x)
			x = Dropout(0.5)(x)
			x = Dense(2, activation='softmax', name='predictions')(x)
		else:
			x = GlobalAveragePooling2D()(base_model.output)
			x = Dropout(0.5)(x)
			x = Dense(2, activation='softmax', name='predictions')(x)
		
		model = Model(base_model.input, x)

		if self.imagenet:
			model.load_weights(self.imagenet)
		
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
		
		if not os.path.exists(self.save_model+self.name_model):
			os.makedirs(self.save_model+self.name_model)
	
		if self.multi_check:
			checkpoint = ModelCheckpoint(
				self.save_model+self.name_model+"/model_epoch{epoch:02d}.hdf5", monitor=mtr, verbose=1, save_best_only=False, mode='max')
		else:
			checkpoint = ModelCheckpoint(
				self.save_model+self.name_model+"/model.hdf5", monitor=mtr, verbose=1, save_best_only=True, mode='max')
			
		earlystop = EarlyStopping(monitor='val_loss',min_delta=0,patience=5,verbose=0, mode='auto')
		
		
		tbCallBack = TensorBoard(
			log_dir='tensorboard/'+self.save_model.replace("models","")+self.name_model,
			histogram_freq=0, write_graph=True, write_images=True
		)
		
		callbacks_list = [tbCallBack, checkpoint]
		
		model.fit_generator(
			self.train_generator,
			epochs=self.epochs,
			validation_data=self.val_generator,
			class_weight=self.weight,
			callbacks=callbacks_list,
			steps_per_epoch=int(self.size_train/self.batch_size),
			validation_steps=int(self.size_val/self.batch_size_val),
			# use_multiprocessing=True,
			max_queue_size=30,
			workers=15
		)

		K.clear_session()
