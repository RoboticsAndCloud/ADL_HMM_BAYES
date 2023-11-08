import random
import numpy as np
from utils import *
from tensorflow.keras.callbacks import TensorBoard
import keras
from keras import models
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import cv2
import time

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential

from keras.regularizers import l2


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
# from keras.layers.advanced_activations import LeakyReLU
from keras.layers import ELU, PReLU, LeakyReLU

from keras.preprocessing.image import ImageDataGenerator

from keras import models

import matplotlib.pyplot as plt
import time

IMG_WIDTH, IMG_HEIGHT = 299, 299 # set this according to keras documentation, each model has its own size
BATCH_SIZE = 50 # decrease this if your computer explodes
MAX_EPOCH = 20

def AlexnetModel1(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),num_classes=6, l2_reg=0.):
    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(96, (11, 11), input_shape=input_shape,
        padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(256, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(256, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(128, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(128, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(128))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(128))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.2))

    # Layer 8
    alexnet.add(Dense(num_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    alexnet.summary()

    learning_rate = 0.001
    batch_size = 128
    alexnet.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    return alexnet

def train():
    #set location of the training data
    datadir = "./watch_data/labelled"
    
    #seperate the images data with categories in data_dir and list them here below
    categories = ["bathroom","bedroom","kitchen", "livingroom", "corridor", "lobby"]
    categories = ['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']
    
    img_size = 299     # resize all the images to one size
    training_data=[]
    create_training_data(categories,datadir,img_size,training_data)
    random.shuffle(training_data)
    X = []
    y = []
    for features,label in training_data:
        X.append(features)
        y.append(label)
    
    X=np.array(X).reshape(-1,img_size,img_size,1)  #(cannot pass list directly, -1=(calculates the array size), size,1=gray scale)
    class_num=keras.utils.np_utils.to_categorical(y,num_classes=len(categories))   #one-hot encoder for cateorical values
    
    X=X/255.0

    model = AlexnetModel1(input_shape = X.shape[1:])
    model.fit(X, class_num, epochs=15, batch_size=32,validation_split=0.2)
    print('model fit complete')

    MODEL_SAVED_PATH = 'watch-saved-model-alex'
    model.save(MODEL_SAVED_PATH)


#train()
#exit(0)

#set location of the training data
datadir = "./watch_data/labelled"

#seperate the images data with categories in data_dir and list them here below
categories = ["bathroom","bedroom","kitchen", "livingroom", "corridor", "lobby"]
categories = ['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']


img_size = 299     # resize all the images to one size
channel = 3
training_data=[]
create_training_data(categories,datadir,img_size,training_data)
random.shuffle(training_data)
X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)

print(len(X))

X=np.array(X).reshape(-1,img_size,img_size,channel)  #(cannot pass list directly, -1=(calculates the array size), size,1=gray scale)
class_num=keras.utils.np_utils.to_categorical(y,num_classes=len(categories))   #one-hot encoder for cateorical values

print('reshape:')
print(len(X))
print(X.ndim)

X=X/255.0

#build model
dense_layers=[4]
layer_sizes=[128]
conv_layers=[2]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            print('----------------------------one')
            name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))
            print(name)
            print('shape:', X.shape[1:])
            tensorboard = TensorBoard(log_dir='.\logs{}'.format(name))
            model = models.Sequential()
            model.add(Conv2D(layer_size,(3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size,(3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(len(categories), activation='softmax'))   
            model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
            model.summary()
            model.fit(X, class_num, epochs=15, batch_size=32,validation_split=0.2,callbacks=[tensorboard])
            print('model fit complete')

            MODEL_SAVED_PATH = 'watch-saved-model'
            model.save(MODEL_SAVED_PATH)

#predicting
#Pass the test image or image for prediction below 
print('executing prediction')
test_img = './Images_test' + '/' + 'hunter_room.jpg'
#test_img = './Images_test' + '/' + 'bedroom.jpg'
test_img = './watch_data/Images_test' + '/' + 'kitchen.jpg'
img_array = cv2.imread(test_img,cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (img_size, img_size))
new_array=new_array.reshape(-1,img_size, img_size,channel)
prediction = model.predict([new_array])
print(prediction)
print(categories[np.argmax(prediction)])
