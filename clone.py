import csv
import cv2
import numpy as np

lines = []
with open('Training_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:

	center_image_path = line[0]
	center_image_filename = center_image_path.split('\\')[-1]
	center_image_full_path = 'Training_data/IMG/' + center_image_filename
	center_image = cv2.imread(center_image_full_path)
	images.append(center_image)
	measurement = float(line[3])
	measurements.append(measurement)

	left_image_path = line[1]
	left_image_filename = left_image_path.split('\\')[-1]
	left_image_full_path = 'Training_data/IMG/' + left_image_filename
	left_image = cv2.imread(left_image_full_path)
	images.append(left_image)
	measurement = float(line[3]) + 0.2
	measurements.append(measurement)

	right_image_path = line[2]
	right_image_filename = right_image_path.split('\\')[-1]
	#print(right_image_filename)
	right_image_full_path = 'Training_data/IMG/' + right_image_filename
	right_image = cv2.imread(right_image_full_path)
	images.append(right_image)
	measurement = float(line[3]) - 0.2
	measurements.append(measurement)


augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((80, 25), (0,0))))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
