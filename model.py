import csv
import cv2
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import scipy.misc
from keras import backend as K 


def cropAndResize(image):
	top = int(np.ceil(image.shape[0] * 0.3))
	bottom = image.shape[0] - int(np.ceil(image.shape[0] * 0.1))
	image = image[top:bottom, :]

	return scipy.misc.imresize(image, (64, 64))
	#return image

def randomize(probability):
	random = randint(0,9)
	if random <= probability:
		return True	
	else:
		return False

def change_brightness(image):
	image_hsv = rgb_to_hsv(image)
	brightness = np.random.uniform() + 0.25
	image_hsv[:,:,2] = image_hsv[:,:,2] * brightness
	return hsv_to_rgb(image_hsv)

# lines = []
# with open('Udacity_data/driving_log.csv') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for line in reader:
# 		lines.append(line)

# images = []
# measurements = []
# for line in lines:

# 	valid = True
# 	center_image_path = line[0]
# 	center_image_filename = center_image_path.split('/')[-1]
# 	center_image_full_path = 'Udacity_data/IMG/' + center_image_filename
# 	center_image = plt.imread(center_image_full_path)
# 	measurement = float(line[3])
# 	if 0.1 > abs(measurement):
# 		if randomize(probability=3):
# 			measurements.append(measurement)
# 			images.append(cropAndResize(center_image))
# 		else:
# 			valid = False
# 	else:
# 		measurements.append(measurement)
# 		images.append(cropAndResize(center_image))

# 	if randomize(probability=10):
# 		left_image_path = line[1]
# 		left_image_filename = left_image_path.split('/')[-1]
# 		left_image_full_path = 'Udacity_data/IMG/' + left_image_filename
# 		left_image = plt.imread(left_image_full_path)
# 		images.append(cropAndResize(left_image))
# 		measurement = float(line[3]) + 0.25
# 		measurements.append(measurement)

# 	if randomize(probability=10):
# 		right_image_path = line[2]
# 		right_image_filename = right_image_path.split('/')[-1]
# 		#print(right_image_filename)
# 		right_image_full_path = 'Udacity_data/IMG/' + right_image_filename
# 		right_image = plt.imread(right_image_full_path)
# 		images.append(cropAndResize(right_image))
# 		measurement = float(line[3]) - 0.25
# 		measurements.append(measurement)


# augmented_images, augmented_measurements = [], []
# for image, measurement in zip(images, measurements):
# 	if randomize(probability=10):
# 		augmented_images.append(change_brightness(image))
# 	else:
# 		augmented_images.append(image)
# 	augmented_measurements.append(measurement)
# 	if randomize(probability=10):
# 		augmented_images.append(np.fliplr(image))
# 		augmented_measurements.append(measurement*-1.0)
# 	# if randomize(probability=7):
# 	# 	image_hsv = rgb_to_hsv(image)
# 	# 	brightness = np.random.uniform() + 0.25
# 	# 	image_hsv[:,:,2] = image_hsv[:,:,2] * brightness
# 	# 	image_rgb = hsv_to_rgb(image_hsv)

# 	# 	augmented_images.append(image_rgb)
# 	# 	augmented_measurements.append(measurement)

# X_train = np.array(images)
# y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import ELU, Dropout, SpatialDropout2D
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers.convolutional import Cropping2D, Convolution2D

def model():
	model = Sequential()
	#model.add(Cropping2D(cropping=((55, 20), (0,0)), input_shape=(64,64,3)))
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64,64,3)))
	model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
	model.add(Dropout(0.1))
	model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
	model.add(Dropout(0.2))
	model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
	model.add(Dropout(0.3))
	model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
	model.add(Dropout(0.2))
	model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
	#model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dropout(0.3))
	model.add(Dense(100, activation="elu"))
	model.add(Dropout(0.5))
	model.add(Dense(50, activation="elu"))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation="elu"))
	model.add(Dropout(0.5))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam')
	return model

def process_full_path(image_name):
	image_name = image_name.split('/')[-1]
	image_name = 'Udacity_data/IMG/' + image_name
	return image_name

def get_image_names_and_labels():
	lines = []
	with open('Udacity_data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	image_names = []
	measurements = []
	for line in lines:
		measurement = float(measurement)
		image_names.append(process_full_path(line[0]), process_full_path(line[1]), process_full_path(line[2]))
		measurements.append(measurement, measurement+0.25, measurement-0.25)

	return image_names, measurements

def batch_generator(X_train, y_train, batch_size=128):
	images = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
	measurements = np.zeros((batch_size,), dtype=np.float32)

	for i in range(batch_size):
		index` = random.randrange(len(X_train))
		image_index = random.randrange(len(X_train[0]))

		measurement = float(y_train[index][image_index])
		image = plt.imread(X_train[index][image_index])

		image = change_brightness(image)
		image = cropAndResize(image)

		if randomize(5):
			image = np.flipr(image)
			measurement = -measurement

		images[i] = image
		measurements[i] = measurement

	yield images, measurements

if __name__=="__main__":
	X_train, y_train = get_image_names_and_labels()
	X_train, y_train = shuffle(X_train, y_train)
	X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

	model = get_model()
	model.summary()
	model.fit_generator(generate_batch(X_train, y_train), samples_per_epoch=10000, nb_epoch=28, validation_data=generate_batch(X_validation, y_validation), nb_val_samples=1024)

	print('Saving model weights and configuration file.')
	model.save_weights('model.h5')
	with open('model.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)

    K.clear_session()
