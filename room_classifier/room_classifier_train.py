import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.xception import Xception, preprocess_input, decode_predictions

import matplotlib.pyplot as plt



# bathroom: 0, bedroom:1, kitchen:2, livingroom:3, lobby:4, door:5
class_names=['bathroom','bedroom', 'kitchen','livingroom', 'lobby', 'door']


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


def create_model(num_classes):
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
    

DIR = "/home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier/room_dataset/"
IMG_WIDTH, IMG_HEIGHT = 299, 299 # set this according to keras documentation, each model has its own size
BATCH_SIZE = 200 # decrease this if your computer explodes

train_generator, validation_generator = create_generators(DIR + "labelled", DIR + "validation")

total_classes = len(train_generator.class_indices) # necesary to build the last softmax layer
to_class = {v:k for k,v in train_generator.class_indices.items()} # usefull when model returns prediction

m = create_model(total_classes)

# Run this several times until you get good acurracy in validation (wachout of overfitting)
# for i in range(MAX_EPOCH):
#     train(train_generator, validation_generator, m)
train2(train_generator, validation_generator, m, epoch=12)



# execute this when you want to save the model
MODEL_SAVED_PATH = 'saved-model2'
m.save(MODEL_SAVED_PATH)

# execute this when you want to load the model
# from keras.models import load_model
# m_i_am_back = load_model(MODEL_SAVED_PATH)




