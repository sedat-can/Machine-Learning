"""CNN model for textual data"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import argparse
import locale
import os
import glob
import cv2

def load_house_attributes(inputPath):
	cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
	zipcodes = df["zipcode"].value_counts().keys().tolist()
	counts = df["zipcode"].value_counts().tolist()
	for (zipcode, count) in zip(zipcodes, counts):
		if count < 25:
			idxs = df[df["zipcode"] == zipcode].index
			df.drop(idxs, inplace=True)
	return df

def process_house_attributes(df, train, test):
	continuous = ["bedrooms", "bathrooms", "area"]
	cs = MinMaxScaler()
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])
	zipBinarizer = LabelBinarizer().fit(df["zipcode"])
	trainCategorical = zipBinarizer.transform(train["zipcode"])
	testCategorical = zipBinarizer.transform(test["zipcode"])
	trainX = np.hstack([trainCategorical, trainContinuous])
	testX = np.hstack([testCategorical, testContinuous])
	return (trainX, testX)

def create_mlp(dim):
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))
	model.add(Dense(1, activation="linear"))
	return model

inputPath = os.path.sep.join(["C:\\Users\\elifh\\OneDrive\\Houses_Dataset","HousesInfo.txt"])
df = load_house_attributes(inputPath)
(train, test) = train_test_split(df, test_size=0.25, random_state=42)

maxPrice = train["price"].max()
trainY = train["price"] / maxPrice
testY = test["price"] / maxPrice

(trainX, testX) = process_house_attributes(df, train, test)

model = create_mlp(trainX.shape[1])
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="MSE", optimizer=opt)
model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=8)
preds = model.predict(testX)

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
error = preds.flatten() - testY
squared_error = error ** 2
MSE = np.mean(squared_error)
r2_score_test = r2_score(testY, preds.flatten())

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
print("MSE = ", MSE)
print("R2_score = ", r2_score_test)

