import numpy as np
# from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
# from keras.models import Model
# from keras.applications.xception import Xception, preprocess_input, decode_predictions
import os
import time
from timeit import default_timer as timer
from PIL import Image


from tflite_runtime.interpreter import Interpreter


import log
import logging


IMG_WIDTH, IMG_HEIGHT = 299, 299 # set this according to keras documentation, each model has its own size
BATCH_SIZE = 200 # decrease this if your computer explodes
TEST_DIR = './test'
TEST_DIR = './room_testset/2/' # kitchen
TEST_DIR = './room_testset/3/' # living

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

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=2):
  set_input_tensor(interpreter, image)

  interpreter.invoke()
  output_details = interpreter.get_output_details()
  # print("output details:", output_details)
  output_details = interpreter.get_output_details()[0]
#   print("output2 details:", output_details)
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  prediction_classes = np.argmax(output)
  print('prediction_classes:', prediction_classes)

  softmax_prob = output/255.0
  print("softmax_prob:", softmax_prob)

  scale, zero_point = output_details['quantization']
  # output = scale * (output - zero_point)
  # print('output:', output)

  # ordered = np.argpartition(-output, 1)
  # print('ordered:', ordered)

  return prediction_classes, softmax_prob[prediction_classes]

# makes the prediction of the file path image passed as parameter 
def predict(file, model, to_class, width, height):

    interpreter = model
    image = Image.open(file).convert('RGB').resize((width, height))

    # Classify the image.
    time1 = time.time()
    label_id, prob = classify_image(interpreter, image)
    print("label id:", label_id)
    time2 = time.time()
    classification_time = np.round(time2-time1, 3)
    print("Classificaiton Time =", classification_time, "seconds.")

    # Read class labels.
    # labels = load_labels(label_path)

    # Return the classification label of the image.
    classification_label = to_class[label_id]
    print("Image Label is :", classification_label, ", with Accuracy :", np.round(prob*100, 2), "%.")
    return to_class[label_id], prob



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
    MODEL_SAVED_PATH = 'saved-model2'

    ml = load_model(MODEL_SAVED_PATH)


    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3, lobby:4, door:5
    class_names=['bathroom','bedroom', 'kitchen','livingroom', 'lobby', 'door']

    import sys, random
    from pathlib import Path
    from PIL import Image
    import matplotlib.pyplot as plt


    # Retreive 9 random images from directory

    files=Path(TEST_DIR).resolve().glob('*.*')
    test_sample= get_file_count_of_dir(TEST_DIR)
    print("test_sample:", test_sample)

    images=random.sample(list(files), test_sample)

    # Configure plots
    # fig = plt.figure(figsize=(9,9))
    # rows,cols = 3,3
    fig = plt.figure(figsize=(36,36))
    rows, cols = 6, 7

    for num,img in enumerate(images):
            file = img
            # print('file:', file)
            start = timer()
            label, prob = predict(file, ml, class_names)

            plt.subplot(rows,cols,num+1)
            plt.title("Pred: "+label + '(' + str(prob) + ')')
            print("Pred: "+label + '(' + str(prob) + ')')
            end = timer()
            print("Time cost:", end - start)
            #plt.axis('off')
            #img = Image.open(img).convert('RGB')
            #plt.imshow(img)
            #plt.savefig("test_res.png")

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
    model_path = "./ascc_efficientNet_model_default.tflite"

    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3, hallway:4, door:5
    labels=['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']
    # label_path = data_folder + "labels_home_v1.txt"
    print("labels:", labels)

    interpreter = Interpreter(model_path)
    print("Model Loaded Successfully.")

    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print("Image Shape (", width, ",", height, ")")
    
    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3, lobby:4, door:5
    class_names=['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']


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
            time.sleep(0.4)
            continue

        if os.path.exists(test_dir) == False:
            print('test_dir not exist:', test_dir)
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
            # label, prob = predict(file, ml, class_names)
            
            label, prob = predict(file, interpreter, class_names, width, height)
            
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



