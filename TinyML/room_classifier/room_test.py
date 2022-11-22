from pydoc import classname
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.xception import Xception, preprocess_input, decode_predictions
import os

IMG_WIDTH, IMG_HEIGHT = 299, 299 # set this according to keras documentation, each model has its own size
IMG_WIDTH, IMG_HEIGHT = 224, 224 # set this according to keras documentation, each model has its own size

BATCH_SIZE = 200 # decrease this if your computer explodes
TEST_DIR = './test'

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
    tmp  = model.predict(x)
    #print('tmp:', tmp)
    index = model.predict(x).argmax()
    return to_class[index]
    

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

def room_sample_plot():
    
    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3, lobby:4, door:5
    class_names=['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']

    sample_folder = './room_samples_0831'


    import sys, random
    from pathlib import Path
    from PIL import Image
    import matplotlib.pyplot as plt


    # Retreive 9 random images from directory

    files=Path(sample_folder).resolve().glob('*.*')
    test_sample= get_file_count_of_dir(sample_folder)

    images=random.sample(list(files), test_sample)

    # Configure plots
    # fig = plt.figure(figsize=(9,9))
    rows,cols = 3,3
    fig = plt.figure(figsize=(12,12))
    # rows, cols = 6, 7

    for num,img in enumerate(images):
            file = img
            print('file:', file.name)
            label = file.name
            label = label.replace('.jpg', '')
            print('label:', label)

            plt.subplot(rows,cols,num+1)
            plt.title(label)
            print("Name: "+label)
            plt.axis('off')
            img = Image.open(img).convert('RGB')
            plt.imshow(img)
            plt.savefig("sample_res.png")


def test():
    # execute this when you want to load the model
    from keras.models import load_model
    MODEL_SAVED_PATH = 'saved-model2'

    ml = load_model(MODEL_SAVED_PATH)
    
    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3, lobby:4, door:5
    class_names=['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']


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
            label = predict(file, ml, class_names)

            plt.subplot(rows,cols,num+1)
            plt.title("Pred: "+label)
            print("Pred: "+label)
            plt.axis('off')
            img = Image.open(img).convert('RGB')
            plt.imshow(img)
            plt.savefig("test_res.png")

def get_confusion_matrix():
    dir = './dataset_online/test/'
    path = dir
    y_test = []
    y_pred = []

    class_names=['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']

    
    for fn in os.listdir(path):
        if os.path.isdir(dir + '/' + fn):
            print(fn)
            test_dir = dir + '/' + fn
            sample_cnt = get_file_count_of_dir(test_dir)
            print('sample cnt:', sample_cnt)
            
            test_truth_label = []
            for i in range(sample_cnt):
                index = int(fn)
                print('test lable:', class_names[index])
                test_truth_label.append(class_names[index])
                

            y_test.extend(test_truth_label)
            print('len test_truth_label:', len(test_truth_label))
            print('len y_test:', len(y_test))

            predict_res = test_confusion_matrix(test_dir)
            print('len(predict_res):', len(predict_res))
            y_pred.extend(predict_res)


    import matplotlib.pyplot as plt
    # confusion matrix
    from mlxtend.plotting import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix

    print('y_pred len:', len(y_pred))
    print('y_pred:', y_pred)
    print('y_test len:', len(y_test))
    print('y_test:', y_test)

    plt.figure()
    mat = confusion_matrix(y_test, y_pred, labels=class_names)
    cm = plot_confusion_matrix(conf_mat=mat, class_names=class_names, show_normed=True, figsize=(7,7))
    plt.show()
    plt.savefig("room_cm_onlinedataset_mnet.png")



def test_confusion_matrix(file_dir):
    from keras.models import load_model
    MODEL_SAVED_PATH = 'saved-model_onlinedataset'

    ml = load_model(MODEL_SAVED_PATH)
    
    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3, lobby:4, door:5
    class_names=['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']


    import sys, random
    from pathlib import Path
    from PIL import Image
    import matplotlib.pyplot as plt


    # Retreive 9 random images from directory

    files=Path(file_dir).resolve().glob('*.*')
    test_sample= get_file_count_of_dir(file_dir)

    images=random.sample(list(files), test_sample)

    # Configure plots
    fig = plt.figure(figsize=(9,9))
    rows,cols = 3,3
    fig = plt.figure(figsize=(36,36))
    rows, cols = 9, 9

    predict_res = []
    cnt = 0

    for num,img in enumerate(images):
           # print("img:", img)
            file = img
            label = predict(file, ml, class_names)
            predict_res.append(label)

            # if label == 'bedroom':
            #     continue

            # # plt.subplot(rows,cols,num+1)
            # plt.subplot(rows,cols,cnt+1)

            # print('file:', file)
            # plt.title("Pred: "+label)
            # print("Pred: "+label)
            # plt.axis('off')
            # img = Image.open(img).convert('RGB')
            # plt.imshow(img)
            # plt.savefig("test_resxx.png")

            # cnt = cnt + 1

            # if cnt > 50:
            #     break

    return predict_res





if __name__ == "__main__":
    print('Test running:===========================================================\n')
    # test()
    # room_sample_plot()
    get_confusion_matrix()
    # test_confusion_matrix('./room_dataset/test/1')




