from sre_parse import CATEGORIES
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

DATADIR = "files"
CATEGORIES = ["bb", "blonde"] #0=black/brown 1=black 2=ginger
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_ANYCOLOR)
                img_array = cv2.resize(img_array,(60,60))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass


create_training_data()
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1,60,60,3)

# pickle_out =open("X.pickle", "wb")
# pickle.dump(X,pickle_out)
# pickle_out.close()

# pickle_out =open("Y.pickle", "wb")
# pickle.dump(Y,pickle_out)
# pickle_out.close()

X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

# X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
            optimizer='adam',
            metrics=['accuracy'])
X = np.array(X)
Y = np.array(Y)
model.fit(X, Y, batch_size=64, epochs=30, validation_split=0.3)

with open("model_pickle", "wb") as f:
    pickle.dump(model,f)



