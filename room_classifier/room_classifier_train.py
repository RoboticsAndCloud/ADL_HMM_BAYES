import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential

from keras.regularizers import l2

from tensorflow import keras
from tensorflow.keras import layers


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

import matplotlib.pyplot as plt



# bathroom: 0, bedroom:1, kitchen:2, livingroom:3, lobby:4, door:5
class_names=['bathroom','bedroom', 'kitchen','livingroom', 'lobby', 'door']

DIR = "/home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier/room_dataset/"
DIR = '/home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier/watch_dataset/Image/'
IMG_WIDTH, IMG_HEIGHT = 299, 299 # set this according to keras documentation, each model has its own size
BATCH_SIZE = 50 # decrease this if your computer explodes

MAX_EPOCH = 20

history = ''

def create_generators(train_data_dir, validation_data_dir):
    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.

    transformation_ratio = .05  # how aggressive will be the data augmentation/transformation

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=transformation_ratio,
                                       shear_range=transformation_ratio,
                                       zoom_range=transformation_ratio,
                                       cval=transformation_ratio,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    
    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                  batch_size=BATCH_SIZE,
                                                                  class_mode='categorical')
    return train_generator, validation_generator

"""
https://keras.io/examples/vision/image_classification_from_scratch/#build-a-model
"""
def make_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), num_classes=6):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    model = keras.Model(inputs, outputs)
    learning_rate = 0.001
    batch_size = 128
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    model.summary()

    return model

def vgg19_model(img_shape=(224, 224, 3), n_classes=1000, l2_reg=0.,
	weights=None):

	# Initialize model
	vgg19 = Sequential()

	# Layer 1 & 2
	vgg19.add(Conv2D(64, (3, 3), padding='same',
		input_shape=img_shape, kernel_regularizer=l2(l2_reg)))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(64, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3 & 4
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(128, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(128, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 5, 6, 7, & 8
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(256, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(256, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(256, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(256, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 9, 10, 11, & 12
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 13, 14, 15, & 16
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 17, 18, & 19
	vgg19.add(Flatten())
	vgg19.add(Dense(4096))
	vgg19.add(Activation('relu'))
	vgg19.add(Dropout(0.5))
	vgg19.add(Dense(4096))
	vgg19.add(Activation('relu'))
	vgg19.add(Dropout(0.5))
	vgg19.add(Dense(n_classes))
	vgg19.add(Activation('softmax'))

	if weights is not None:
		vgg19.load_weights(weights)

	return vgg19

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
    alexnet.add(Conv2D(512, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    
    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))
    
    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))
    
    # Layer 8
    alexnet.add(Dense(num_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))
    
    alexnet.summary()
    
    learning_rate = 0.001
    batch_size = 128
    alexnet.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    
    return alexnet
    

def AlexnetModel(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),num_classes=6):
    mobile_net = MobileNet(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False,
        pooling='max'
    )

#    efficient_net = EfficientNetB3(
#        weights='imagenet',
#        input_shape=input_shape,
#        include_top=False,
#        pooling='max'
#    )    
    
    model = Sequential()
    model.add(mobile_net)
    model.summary()

    #model = Sequential()
    #model.add(Conv2D(filters=96,kernel_size=(3,3),strides=(4,4),input_shape=input_shape, activation='relu'))
    #model.add(Conv2D(filters=96,kernel_size=(3,3),strides=(4,4), activation='relu'))
    model.add(Flatten())
    model.add(Conv2D(64,(5,5),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(256,(5,5),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes,activation='softmax'))
        
    learning_rate = 0.001
    batch_size = 128
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
        #model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        
    model.summary()
    return model

def create_model(num_classes):
    # Define models 
    
    model = Sequential()
    
    model.add(Conv2D(6, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    BatchNormalization(axis=-1)
    
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(32,(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(82, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    
    model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    
    # Flattening is a technique that is used to convert multi-dimensional arrays into a 1-D array
    model.add(Flatten())
    model.add(BatchNormalization(axis=-1))
    
    # Fully connected layer
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6))
    
    model.add(Activation('softmax'))
    
    print("Layers:", len(model.layers))
    print("Layers:", model.layers)
    model.summary()
    

    learning_rate = 0.001
    batch_size = 128
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    return model
    




def create_model_transferlearning(num_classes):
        base_model = Xception(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), weights='imagenet', include_top=False)

        # Top Model Block
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        # add your top layer block to your base model
        model = Model(base_model.input, predictions)
        
        for layer in model.layers[:-10]:
            layer.trainable = False
        
        model.compile(optimizer='nadam',loss='categorical_crossentropy', metrics=['accuracy'])
        return model


def plot_learningCurve(history, epochs):
  # Plot training & validation accuracy values
  plt.figure()
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()
  plt.savefig("accuracy.png")

  # Plot training & validation loss values
  plt.figure()
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()
  plt.savefig("loss.png")


def train2(train_generator, validation_generator, model, epoch = 1):
    global history
    history = model.fit_generator(train_generator,
                        epochs=epoch,
                        validation_data=validation_generator,
                        steps_per_epoch=3,
                        validation_steps=2,
                        verbose=1)

    #print(history.history['accuracy'])
    #print(history.history['val_accuracy'])
    
    plot_learningCurve(history, epoch)


def train(train_generator, validation_generator, model):
    model.fit_generator(train_generator,
                        epochs=1,
                        validation_data=validation_generator,
                        steps_per_epoch=3,
                        validation_steps=2,
                        verbose=1)

# makes the prediction of the file path image passed as parameter 
def predict(file, model, to_class):
    im = load_img(file, target_size=(IMG_WIDTH, IMG_HEIGHT))
    x = img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    index = model.predict(x).argmax()
    return to_class[index]
    


train_generator, validation_generator = create_generators(DIR + "labelled", DIR + "validation")

total_classes = len(train_generator.class_indices) # necesary to build the last softmax layer
to_class = {v:k for k,v in train_generator.class_indices.items()} # usefull when model returns prediction

m = AlexnetModel1() 
#m = make_model()
#m = AlexnetModel() 
#m = create_model(total_classes)
#m = create_model_transferlearning(total_classes)

# Run this several times until you get good acurracy in validation (wachout of overfitting)
# for i in range(MAX_EPOCH):
#     train(train_generator, validation_generator, m)
train2(train_generator, validation_generator, m, epoch=150)



# execute this when you want to save the model
MODEL_SAVED_PATH = 'watch-saved-model'
m.save(MODEL_SAVED_PATH)

# execute this when you want to load the model
# from keras.models import load_model
# m_i_am_back = load_model(MODEL_SAVED_PATH)




