import tensorflow as tf
import numpy as np
import cv2
import os,random
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

class myc(tf.keras.callbacks.Callback):
	def on_epoch_end(self,epoch,logs={}):
		if logs.get('loss')<0.01:
			print('HI! desired accuracy reached so cancelling training ')
			self.model.stop_training=True


#creating directories for training and testing
created =[
"/maskvsnomask",
"/maskvsnomask/training",
"/maskvsnomask/testing",
"/maskvsnomask/training/masks",
"/maskvsnomask/training/nomasks",
"/maskvsnomask/testing/masks",
"/maskvsnomask/testing/nomasks"


]
for d in created:
	try:
		os.mkdir(d)
		print(f"{d} successfully created" )
	except:
		print(f'{d} creation failed! ')

def split_data(SOURCE,training,testing,split_size):
	all_files=[]
	for file_name in os.listdir(SOURCE):
		file_path=SOURCE+file_name
		if os.path.getsize(file_path):
			all_files.append(file_name)
		else:
			print(f"{file_name} is corrupt! so ignoring it" )

	n=len(all_files)
	split_point=int(split_size*n)
	shuffled=random.sample(all_files,n)
	train_set=shuffled[:split_point]
	test_set=shuffled[split_point:]

	for file_name in train_set:
		copyfile(SOURCE + file_name, training + file_name )

	for file_name in test_set:
		copyfile(SOURCE + file_name, testing + file_name )

split_data("C:/Users/DELL/Desktop/AI/Face detector/dataset/with mask/","/maskvsnomask/training/masks/","/maskvsnomask/testing/masks/",0.9)
split_data("C:/Users/DELL/Desktop/AI/Face detector/dataset/without mask/","/maskvsnomask/training/nomasks/","/maskvsnomask/testing/nomasks/",0.9)

print(len(os.listdir("/maskvsnomask/training/masks/")))
print(len(os.listdir("/maskvsnomask/testing/masks/")))
print(len(os.listdir("/maskvsnomask/training/nomasks/")))
print(len(os.listdir("/maskvsnomask/testing/masks/")))

callbacks=myc()			
model=tf.keras.Sequential([

tf.keras.layers.Conv2D(256,(3,3),activation='relu',input_shape= (100,100,1)),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dropout(0.4),
tf.keras.layers.Dense(512,activation='relu'),
tf.keras.layers.Dense(128,activation='relu'),
tf.keras.layers.Dense(64,activation='relu'),
tf.keras.layers.Dense(2,activation='softmax')
])
train_datagen=ImageDataGenerator(rescale = 1./255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.25,
	horizontal_flip=True,
	fill_mode='nearest'
	)
training_generator= train_datagen.flow_from_directory(
"C:/maskvsnomask/training",
target_size=(100,100),
batch_size=10,
color_mode='grayscale',
class_mode='categorical')

validation_datagen=ImageDataGenerator(rescale=1./255)
validation_generator= train_datagen.flow_from_directory(
"C:/maskvsnomask/testing/",
target_size=(100,100),
batch_size=10,
color_mode='grayscale',
class_mode='categorical')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit_generator(training_generator,epochs=3,callbacks=[checkpoint],verbose=1,validation_data=validation_generator)
model.predict("1-with-mask.jpg")














