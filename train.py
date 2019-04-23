import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import model as mdl
import keras
import random
import pickle
import ntpath
import cv2

from PIL import Image
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

data_dir = "track"

def path_file(path):
	""" Parse the filename out of a given path """
	head, tail = ntpath.split(path)
	return tail

def load_img_steering(data):
	""" Parse images and their corresponding steering angles """
	img_path = []
	steering = []
	for i in range(len(data)):
		indexed_data = data.iloc[i]
		center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
		img_path.append(data_dir + "/IMG/" + center.strip())
		steering.append(float(indexed_data[3]))
	img_path = np.asarray(img_path)
	steering = np.asarray(steering)
	return img_path, steering

def aug_zoom(image):
	""" Apply zoom augmentation to the image """
	zoom = iaa.Affine(scale=(1, 1.3))
	return zoom.augment_image(image)

def aug_pan(image):
	""" Apply pan augmentation to the image """
	pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
	return pan.augment_image(image)

def aug_brightness(image):
	""" Apply brightness augmentation to the image """
	brightness = iaa.Multiply((0.2, 1.2))
	return brightness.augment_image(image)

def aug_flip(image, steering_angle):
	""" Flips the image and its steering_angle """
	image = cv2.flip(image, 1)
	steering_angle = -steering_angle
	return image, steering_angle

def augment_random(image_path, steering_angle):
	""" Applies augmentation into an image based on pure luck """
	img = mpimg.imread(image_path)
	steering = steering_angle
	if np.random.rand() < 0.5:
		img = aug_pan(img)
	if np.random.rand() < 0.5:
		img = aug_zoom(img)
	if np.random.rand() < 0.5:
		img = aug_brightness(img)
	if np.random.rand() < 0.5:
		img, steering = aug_flip(img, steering_angle)
	return img, steering

def batch_generator(image_paths, steering_angles, batch_size, istraining):
	""" Batch generator function -- Applies random augmentations into images while training """
	while True:
		batch_images = []
		batch_steerings = []
		for i in range(batch_size):
			random_index = random.randint(0, len(image_paths) - 1)
			if istraining:
				img, steering = augment_random(image_paths[random_index], steering_angles[random_index])
			else:
				img, steering = mpimg.imread(image_paths[random_index]), steering_angles[random_index]
			
			batch_images.append(mdl.image_process(img))
			batch_steerings.append(steering)
		yield (np.asarray(batch_images), np.asarray(batch_steerings))


def preview_training_data(y_train, y_valid, num_bins):
	""" Preview the training data distribution """
	fig, axes = plt.subplots(1, 2, figsize=(12, 4))
	axes[0].hist(y_train, bins=num_bins, width=0.05, color="blue")
	axes[0].set_title("Training set")

	axes[1].hist(y_valid, bins=num_bins, width=0.05, color="red")
	axes[1].set_title("Validation set")

	plt.show()

def preview_image(img_path):
	""" Preview an image before and after training preprosessing """
	image = mpimg.imread(img_path)
	fig, axes = plt.subplots(1, 2, figsize=(15, 10))
	fig.tight_layout()
	axes[0].imshow(image)
	axes[0].set_title("Original Image")
	image = mdl.image_process(image)
	axes[1].imshow(image)
	axes[1].set_title("New Image")
	plt.show()

# Load training data
columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
data = pd.read_csv(data_dir + "/driving_log.csv", names=columns)
pd.set_option("display.max_colwidth", -1)

# Get the file name from full path
data["center"] = data["center"].apply(path_file)
data["right"] = data["right"].apply(path_file)
data["left"] = data["left"].apply(path_file)

num_bins = 25
samples_per_bin = 300

hist, bins = np.histogram(data["steering"], (num_bins))

remove_list = []
for j in range(num_bins):
	lst = []
	for i in range(len(data["steering"])):
		if data["steering"][i] >= bins[j] and data["steering"][i] <= bins[j+1]:
			lst.append(i)
		# The current training set has a preponderance of right turns. Lets manually balance it out
		# if data["steering"][i] >= bins[len(bins) - 1]: # and random.randint(0, 1) == 1:
		# 	lst.append(i)
	lst = shuffle(lst)
	lst = lst[samples_per_bin:]
	remove_list.extend(lst)

# Even out the distribution
data.drop(data.index[remove_list], inplace=True)

img_paths, steerings = load_img_steering(data)

x_train, x_valid, y_train, y_valid = train_test_split(img_paths, steerings, test_size=0.2, random_state=56)

print("training samples: " + str(len(x_train)))
print("validation samples: " + str(len(x_valid)))

# Preview data distribution
preview_training_data(y_train, y_valid, num_bins)

# Create a model
model = mdl.model_create()

# Train it!
history = mdl.model_train_with_generator(model, batch_generator, x_train, y_train, x_valid, y_valid)

# Plot training loss
mdl.model_plot_loss(history)
