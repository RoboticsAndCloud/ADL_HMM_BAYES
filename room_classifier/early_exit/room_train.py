import random
import numpy as np
from utils import *
from tensorflow.keras.callbacks import TensorBoard
import keras
from keras import models
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import cv2
import time

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
new_array=new_array.reshape(-1,img_size, img_size,1)
prediction = model.predict([new_array])
print(prediction)
print(categories[np.argmax(prediction)])
