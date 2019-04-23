from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Convolution2D

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2

def model_save(mdl):
	""" Saves the model to disk """
	mdl.save("model.h5")

def model_load():
	""" Loads the model from disk """
	return load_model("model.h5")

def model_create():
	""" Create an NVIDIA model """
	mdl = Sequential()

	mdl.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation="elu"))
	mdl.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="elu"))
	mdl.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="elu"))
	mdl.add(Convolution2D(64, 3, 3, activation="elu"))
	mdl.add(Convolution2D(64, 3, 3, activation="elu"))

	mdl.add(Flatten())
	mdl.add(Dense(100, activation="elu"))

	mdl.add(Dense(50, activation="elu"))
	mdl.add(Dense(10, activation="elu"))
	mdl.add(Dense(1))

	mdl.compile(optimizer=Adam(lr=1e-4), loss="mse")
	return mdl

def model_train(model, x_train, y_train, x_val, y_val):
	""" Train the model with a single set of data """
	h = model.fit(x_train, y_train, batch_size=100, epochs=30, validation_data=(x_val, y_val), verbose=1, shuffle=1)
	model_save(model)
	return h

def model_train_with_generator(model, datagen, x_train, y_train, x_val, y_val):
	""" Train the model with a single dataset + an image generator """
	h = model.fit_generator(datagen(x_train, y_train, 100, 1), steps_per_epoch=300,
		epochs=10, validation_data=datagen(x_val, y_val, 100, 0), validation_steps=200,
		verbose=1, shuffle=1)
	model_save(model)
	return h

def model_plot_loss(h):
	""" Plot model loss """
	plt.plot(h.history["loss"])
	plt.plot(h.history["val_loss"])
	plt.legend(["loss", "val_loss"])
	plt.title("Loss")
	plt.xlabel("epoch")
	plt.show()

def model_plot_acc(h):
	""" Plot model accuracy """
	plt.plot(h.history["acc"])
	plt.plot(h.history["val_acc"])
	plt.legend(["acc", "val_acc"])
	plt.title("Accuracy")
	plt.xlabel("epoch")
	plt.show()

def image_process(img):
	""" Preprocess the image by resizing + applying YUV color schema & blur  """
	img = img[60:135, :, :]
	img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
	img = cv2.GaussianBlur(img, (3, 3), 0)
	img = cv2.resize(img, (200, 66))
	return img / 255

def image_process_path(img_path):
	""" Read the image from specified path and run it through the preprocessor """
	return image_process(mpimg.imread(img_path))
