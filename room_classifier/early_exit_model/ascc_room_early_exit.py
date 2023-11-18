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

import tensorflow as tf
# utils
import pickle


EPOCHS = 15


class ExitBranch(tf.keras.layers.Layer):
    "Exit classifier branch"
    def __init__(self, num_classes):
      super(ExitBranch, self).__init__()
      self.classifier = tf.keras.Sequential([
        # flattening (global pooling)
        tf.keras.layers.GlobalAvgPool2D(),
        # classifier
        tf.keras.layers.Dense(num_classes, activation='softmax')
      ])
      
    def call(self, x, training=False):
      logits = self.classifier(x, training = training)
      return logits


IMG_WIDTH, IMG_HEIGHT = 299, 299 # set this according to keras documentation, each model has its own size
BATCH_SIZE = 50 # decrease this if your computer explodes
MAX_EPOCH = 20

img_size = 299     # resize all the images to one size
#img_size = 32     # resize all the images to one size


#set location of the training data
datadir = "./watch_data/labelled"

#seperate the images data with categories in data_dir and list them here below
categories = ["bathroom","bedroom","kitchen", "livingroom", "corridor", "lobby"]
categories = ['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']


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

print('reshape: len X {}, len(class_num) {}', len(X), len(class_num))
print(len(X))
print(len(class_num))
print(X.ndim)

X=X/255.0

#build model
dense_layer=4
layer_size=128
conv_layer=2


print('----------------------------one')
name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))
print(name)
print('shape:', X.shape[1:])
tensorboard = TensorBoard(log_dir='./log/logs{}'.format(name))


# define callback: Early Stopping 
early_stopping_callback  = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                                                            patience = 10,
                                                            restore_best_weights = True, 
                                                            verbose = 1)

def modelFull(model_name='full'):
  model = models.Sequential()
  model.add(Conv2D(layer_size,(3,3), input_shape = X.shape[1:]))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  for l in range(conv_layer):
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
  # model.fit(X, class_num, epochs=15, batch_size=32,validation_split=0.2,callbacks=[tensorboard])


    # ---------- TRAINING --------------------
  tf.keras.utils.set_random_seed(1)
  with tf.device('/GPU:0'):
      history = model.fit(X, class_num, epochs=EPOCHS, batch_size=32,validation_split=0.2,callbacks=[early_stopping_callback])
  # ---------- SAVE -------------------------
  with open(f'history_exitVggnet11_training1.pkl'+model_name, 'wb') as f:
      pickle.dump(history.history, f)
  print('model fit complete')

  MODEL_SAVED_PATH = 'watch-saved-model-' + model_name
  model.save(MODEL_SAVED_PATH)


  return model


def modelExit1(model_name='exit1'):
  model = models.Sequential()
  model.add(Conv2D(layer_size,(3,3), input_shape = X.shape[1:]))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  for l in range(conv_layer-1):
      model.add(Conv2D(layer_size,(3,3)))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size=(2,2)))
  
  exit_layer = ExitBranch(num_classes=len(categories))
  model.add(exit_layer)

  # logits_e = exit_e(x, training=training)
   

  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  model.summary()
  # model.fit(X, class_num, epochs=15, batch_size=32,validation_split=0.2,callbacks=[tensorboard])
  # print('model fit complete')

  # MODEL_SAVED_PATH = 'watch-saved-model-' + model_name
  # model.save(MODEL_SAVED_PATH)

  # ---------- TRAINING --------------------
  tf.keras.utils.set_random_seed(1)
  with tf.device('/GPU:0'):
      history = model.fit(X, class_num, epochs=EPOCHS, batch_size=32,validation_split=0.2,callbacks=[early_stopping_callback])
  # ---------- SAVE -------------------------
  with open(f'history_exitVggnet11_training1.pkl'+model_name, 'wb') as f:
      pickle.dump(history.history, f)
  print('model fit complete')

  MODEL_SAVED_PATH = 'watch-saved-model-' + model_name
  model.save(MODEL_SAVED_PATH)
  return model


def modelExit2(model_name='exit2'):
  model = models.Sequential()
  model.add(Conv2D(layer_size,(3,3), input_shape = X.shape[1:]))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  for l in range(conv_layer):
      model.add(Conv2D(layer_size,(3,3)))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size=(2,2)))

  #model.add(Flatten())
  #model.add(Dense(layer_size, activation='relu'))
  #model.add(Dense(len(categories), activation='softmax'))   
  exit_layer = ExitBranch(num_classes=len(categories))
  model.add(exit_layer)
   

  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  model.summary()

  # ---------- TRAINING --------------------
  tf.keras.utils.set_random_seed(1)
  with tf.device('/GPU:0'):
      history = model.fit(X, class_num, epochs=EPOCHS, batch_size=32,validation_split=0.2,callbacks=[early_stopping_callback])
  # ---------- SAVE -------------------------
  with open(f'history_exitVggnet11_training1.pkl'+model_name, 'wb') as f:
      pickle.dump(history.history, f)
  print('model fit complete')

  MODEL_SAVED_PATH = 'watch-saved-model-' + model_name
  model.save(MODEL_SAVED_PATH)
  return model



model = modelFull()
#model = modelExit1()
#model = modelExit2()


# define parameters for the plotting
params_dict = {"axes.titlesize" : 24, "axes.labelsize" : 20,
               "lines.linewidth" : 4, "lines.markersize" : 10,
               "xtick.labelsize" : 16,"ytick.labelsize" : 16}
# load the history
with open('history_vggnet11_base.pkl', 'rb') as f:
    history = pickle.load(f)
# Learning curves
with plt.rc_context(params_dict):
    fig, ax = plt.subplots(1, 2, figsize = (20, 7))

    fig.suptitle("Learning curves of the model", fontsize = 25)
    ax[0].plot(history['loss'], label = "train", color = "blue")
    ax[0].plot(history['val_loss'], label = "val", color = "red")
    ax[0].legend(fontsize = 20)
    ax[0].set_xlabel("Epochs"); ax[0].set_ylabel("Loss")

    ax[1].plot(history['sparse_categorical_accuracy'], label = "train", color = "blue")
    ax[1].plot(history['val_sparse_categorical_accuracy'], label = "val", color = "red")
    ax[1].legend(fontsize = 20)
    ax[1].set_xlabel("Epochs"); ax[1].set_ylabel("Accuracy")
    plt.show()


#predicting
#Pass the test image or image for prediction below 
print('executing prediction')
test_img = './Images_test' + '/' + 'hunter_room.jpg'
#test_img = './Images_test' + '/' + 'bedroom.jpg'
test_img = './watch_data/Images_test' + '/' + 'kitchen.jpg'
test_img = './watch_data/Images_test' + '/' + 'bedroom.jpg'
#img_array = cv2.imread(test_img,cv2.IMREAD_GRAYSCALE)
img_array = cv2.imread(test_img,cv2.IMREAD_COLOR)
new_array = cv2.resize(img_array, (img_size, img_size))
new_array=new_array.reshape(-1,img_size, img_size,channel)
prediction = model.predict([new_array])
print(prediction)
print(categories[np.argmax(prediction)])
