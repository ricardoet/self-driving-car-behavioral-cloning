import csv
import cv2
import numpy as np
from random import randint

def cropAndResizeImage(imageIn):
    cropped = imageIn[50:25, 0:360]
    resized = cv2.resize(cropped,(64,64))
    return cropped

lines = []
with open('Training_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:

	valid = True
	center_image_path = line[0]
	center_image_filename = center_image_path.split('\\')[-1]
	center_image_full_path = 'Training_data/IMG/' + center_image_filename
	center_image = cv2.imread(center_image_full_path)
	images.append(center_image)
	measurement = float(line[3])
	if measurement == float(0):
		random = randint(0,9)
		if random >= 3:
			valid = False

	if valid:
		measurements.append(measurement)

		random = randint(0,9)
		if random <= 4:
			left_image_path = line[1]
			left_image_filename = left_image_path.split('\\')[-1]
			left_image_full_path = 'Training_data/IMG/' + left_image_filename
			left_image = cv2.imread(left_image_full_path)
			images.append(left_image)
			measurement = float(line[3]) + 0.25
			measurements.append(measurement)

		random = randint(0,9)
		if random <= 4:
			right_image_path = line[2]
			right_image_filename = right_image_path.split('\\')[-1]
			#print(right_image_filename)
			right_image_full_path = 'Training_data/IMG/' + right_image_filename
			right_image = cv2.imread(right_image_full_path)
			images.append(right_image)
			measurement = float(line[3]) - 0.25
			measurements.append(measurement)


augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

X_train = cropAndResizeImage(X_train)

from keras.models import Sequential
from keras.layers import ELU, Dropout, SpatialDropout2D
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers.convolutional import Cropping2D, Convolution2D

# model = Sequential()
# model.add(Cropping2D(cropping=((80, 25), (0,0)), input_shape=(160,320,3)))
# model.add(Lambda(lambda x: x / 127.5 - 1))

# model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(Flatten())
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Dense(512))
# model.add(Dropout(.5))
# model.add(ELU())
# model.add(Dense(1))

model = Sequential()
#model.add(Cropping2D(cropping=((50, 25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5), input_shape=(64,64,3))
model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
model.add(SpatialDropout2D(0.4))
model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
model.add(SpatialDropout2D(0.4))
model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
model.add(SpatialDropout2D(0.4))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
model.add(SpatialDropout2D(0.4))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
model.add(SpatialDropout2D(0.4))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation="elu"))
model.add(Dense(50, activation="elu"))
model.add(Dense(10, activation="elu"))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')
