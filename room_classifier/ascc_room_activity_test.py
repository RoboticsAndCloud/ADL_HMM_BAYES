import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.xception import Xception, preprocess_input, decode_predictions
import os
import time
from timeit import default_timer as timer


import log
import logging


IMG_WIDTH, IMG_HEIGHT = 299, 299 # set this according to keras documentation, each model has its own size
BATCH_SIZE = 200 # decrease this if your computer explodes
TEST_DIR = './test'

ASCC_DATA_NOTICE_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/ascc_data/notice.txt'
ASCC_DATA_RES_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/ascc_data/recognition_result.txt'

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


def train(train_generator, validation_generator, model):
    model.fit_generator(train_generator,
                        epochs=1,
                        validation_data=validation_generator,
                        steps_per_epoch=3,
                        validation_steps=2,
                        verbose=1)

"""
Brief: get file count of a director

Raises:
     NotImplementedError
     FileNotFoundError
"""
def get_file_count_of_dir(dir, prefix=''):
    path = dir
    count = 0
    for fn in os.listdir(path):
        if os.path.isfile(dir + '/' + fn):
            if prefix != '':
                if prefix in fn:
                    count = count + 1
            else:
                count = count + 1
        else:
            print('fn:', fn)
    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count


# makes the prediction of the file path image passed as parameter 
def predict(file, model, to_class):
    im = load_img(file, target_size=(IMG_WIDTH, IMG_HEIGHT))
    x = img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prob  = model.predict(x)
    print('prob:', prob)
    index = model.predict(x).argmax()
    res_prob = prob[0][index]
    return to_class[index], res_prob
    

# DIR = "./"
# IMG_WIDTH, IMG_HEIGHT = 299, 299 # set this according to keras documentation, each model has its own size
# BATCH_SIZE = 200 # decrease this if your computer explodes

# train_generator, validation_generator = create_generators(DIR + "labelled", DIR + "validation")

# total_classes = len(train_generator.class_indices) # necesary to build the last softmax layer
# to_class = {v:k for k,v in train_generator.class_indices.items()} # usefull when model returns prediction

# m = create_model(total_classes)

# # Run this several times until you get good acurracy in validation (wachout of overfitting)
# train(train_generator, validation_generator, m)
# train(train_generator, validation_generator, m)

# # execute this when you want to save the model
# MODEL_SAVED_PATH = 'saved-model'
# m.save(MODEL_SAVED_PATH)


def test():
    # execute this when you want to load the model
    from keras.models import load_model
    MODEL_SAVED_PATH = 'saved-model_12_lobby'

    ml = load_model(MODEL_SAVED_PATH)
    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3
    # Chores:4, Desk_activity:5, Dining_room_activity:6, Eve_Med:7, Leaving_home: 8, Meditate:9, Morning_Med:10, Reading: 11
    
    # !!!Notice: the list of class_names should correspond to the list of the result of 'ls ./labelled', like '0  1  10  11  2  3  4  5  6  7  8  9'
    class_names=['bathroom','bedroom', 'morning_med', 'reading', 'kitchen','livingroom', 'chores', 'desk_activity', 'dining_room_activity',
                 'eve_med', 'leaving_home', 'meditate']

    class_names=['bathroom','bedroom', 'morning_med', 'reading', 'kitchen','livingroom', 'chores', 'desk_activity', 'dining_room_activity',
                 'eve_med', 'leaving_home', 'lobby']

    import sys, random
    from pathlib import Path
    from PIL import Image
    import matplotlib.pyplot as plt


    # Retreive 9 random images from directory

    files=Path(TEST_DIR).resolve().glob('*.*')
    test_sample= get_file_count_of_dir(TEST_DIR)

    images=random.sample(list(files), test_sample)

    # Configure plots
    # fig = plt.figure(figsize=(9,9))
    # rows,cols = 3,3
    fig = plt.figure(figsize=(36,36))
    rows, cols = 6, 7

    for num,img in enumerate(images):
            file = img
            # print('file:', file)
            label, prob = predict(file, ml, class_names)

            plt.subplot(rows,cols,num+1)
            plt.title("Pred: "+label + '(' + str(prob) + ')')
            print("Pred: "+label + '(' + str(prob) + ')')
            plt.axis('off')
            img = Image.open(img).convert('RGB')
            plt.imshow(img)
            plt.savefig("test_res.png")

def read_dir_name(file_name):
    with open(file_name, 'r') as f:
        dir_name = str(f.read().strip())
        f.close()
    print('dir_name:%s', dir_name)
    return dir_name

def write_res_into_file(file_name, res_list):
    with open(file_name, 'w') as f:
        for v in res_list:
            f.write(str(v))
            f.write('\t')
        f.close()
    
    return True

def run():
    # execute this when you want to load the model
    from keras.models import load_model
    MODEL_SAVED_PATH = 'saved-model_12_lobby'

    ml = load_model(MODEL_SAVED_PATH)
    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3
    # Chores:4, Desk_activity:5, Dining_room_activity:6, Eve_Med:7, Leaving_home: 8, Meditate:9, Morning_Med:10, Reading: 11
    
    # !!!Notice: the list of class_names should correspond to the list of the result of 'ls ./labelled', like '0  1  10  11  2  3  4  5  6  7  8  9'
    class_names=['bathroom','bedroom', 'morning_med', 'reading', 'kitchen','livingroom', 'chores', 'desk_activity', 'dining_room_activity',
                 'eve_med', 'leaving_home', 'meditate']

    class_names=['bathroom','bedroom', 'morning_med', 'reading', 'kitchen','livingroom', 'chores', 'desk_activity', 'dining_room_activity',
                 'eve_med', 'leaving_home', 'lobby']

    import sys, random
    from pathlib import Path
    from PIL import Image
    import matplotlib.pyplot as plt


    # Retreive 9 random images from directory
    
    pre_test_dir = ''
    while True:
        test_dir = read_dir_name(ASCC_DATA_NOTICE_FILE)
        # logging.info('pre_test_dir:%s', pre_test_dir)
        logging.info('got cur test_dir:%s', test_dir)

        if pre_test_dir == test_dir:
            time.sleep(1)
            continue

        pre_test_dir = test_dir

        files=Path(test_dir).resolve().glob('*.*')

        test_sample= get_file_count_of_dir(test_dir) 
        #test_sample= 5

        images=random.sample(list(files), test_sample)

        # Configure plots
        # fig = plt.figure(figsize=(9,9))
        # rows,cols = 3,3
        fig = plt.figure(figsize=(36,36))
        rows, cols = 6, 7

        res = []
        start = timer()
        
        for num,img in enumerate(images):
            file = img
            # print('file:', file)
            label, prob = predict(file, ml, class_names)

            # plt.subplot(rows,cols,num+1)
            # plt.title("Pred: "+label + '(' + str(prob) + ')')
            print("Pred: "+label + '(' + str(prob) + ')')

            logging.info('cur test_dir:%s', test_dir)
            logging.info('Pred:%s', label + '(' + str(prob) + ')')

            res.append(label + '(' + str(prob) + ')')
            # plt.axis('off')
            # img = Image.open(img).convert('RGB')
            # plt.imshow(img)
            # plt.savefig("test_res.png")

        end = timer()
        print("Get_prediction time cost:", end-start)    
        
        write_res_into_file(ASCC_DATA_RES_FILE, res)            


if __name__ == "__main__":
    print('Test running:===========================================================\n')

    log.init_log("./log/my_program")  # ./log/my_program.log./log/my_program.log.wf7
    logging.info("Hello World!!!")
    # test()
    run()




