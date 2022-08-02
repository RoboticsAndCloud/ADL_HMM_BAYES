import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.xception import Xception, preprocess_input, decode_predictions
import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import scipy.stats as stats


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
    MODEL_SAVED_PATH = 'moition-saved-model'

    model = load_model(MODEL_SAVED_PATH)
    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3
    # Chores:4, Desk_activity:5, Dining_room_activity:6, Eve_Med:7, Leaving_home: 8, Meditate:9, Morning_Med:10, Reading: 11
    
    # !!!Notice: the list of class_names should correspond to the list of the result of 'ls ./labelled', like '0  1  10  11  2  3  4  5  6  7  8  9'
    class_names=['bathroom','bedroom', 'morning_med', 'reading', 'kitchen','livingroom', 'chores', 'desk_activity', 'dining_room_activity',
                 'eve_med', 'leaving_home', 'meditate']

    X_test = [[
  [[ 5.03429609e-04],
   [ 1.00909774e+00],
   [-1.28581971e+00],],
  [[ 4.07010313e-02],
   [ 3.69590441e-01],
   [ 2.67269293e-01]],

  [[-4.70904762e-01],
   [-1.07021439e+00],
   [ 2.96503914e-01]],

  [[ 8.08986222e-02],
   [ 2.16108732e-01],
   [ 1.02824588e-01]],

  [[ 7.89838055e-01],
   [-1.34794321e+00],
   [ 2.96503914e-01]],

  [[-4.12435519e-01],
   [-5.33028318e-01],
   [-1.29312837e+00]],

  [[-1.52978303e-01],
   [ 7.09442874e-01],
   [-1.74904269e-01]],

  [[-7.25831130e-02],
   [ 8.73887686e-01],
   [-7.92485516e-01]],

  [[-1.99475917e+00],
   [ 9.03122173e-01],
   [ 2.05145736e-01]],

  [[-7.25831130e-02],
   [-7.08436020e-01],
   [ 1.19181407e+00]],

  [[ 6.69245229e-01],
   [ 4.20751055e-01],
   [-2.37027830e-01]],

  [[ 6.69245229e-01],
   [-7.25831130e-02],
   [-1.74904269e-01]],

  [[ 1.43022206e-01],
   [ 2.05145736e-01],
   [-4.01472549e-01]],

  [[-1.85867249e-01],
   [-1.34706670e-01],
   [ 1.02824588e-01]],

  [[ 4.71911615e-01],
   [ 2.16108732e-01],
   [-6.68238429e-01]],

  [[-1.52978303e-01],
   [-5.95151902e-01],
   [ 2.67269293e-01]],

  [[-1.42833845e+00],
   [ 1.75911115e-01],
   [-3.96941774e-02]],

  [[ 1.75911115e-01],
   [-4.70904762e-01],
   [-1.34706670e-01]],

  [[-7.70559577e-01],
   [ 2.24293902e-02],
   [ 6.26269973e-02]],

  [[-3.68583600e-01],
   [-1.23743690e-01],
   [ 2.67269293e-01]],

  [[ 1.14664180e-02],
   [ 4.31713998e-01],
   [-2.04138885e-01]],

  [[-7.70559577e-01],
   [-3.68583600e-01],
   [ 2.45343354e-01]],

  [[-1.34706670e-01],
   [-3.61274948e-01],
   [ 2.05145736e-01]],

  [[-1.63941289e-01],
   [ 6.69245229e-01],
   [-1.42015325e-01]],

  [[ 2.78232289e-01],
   [-1.42015325e-01],
   [ 2.24293902e-02]],

  [[-5.06571470e-02],
   [ 1.13787558e-01],
   [ 2.24293902e-02]],

  [[-2.14225257e-02],
   [-2.88188408e-01],
   [-4.36660293e-01]],

  [[ 1.25334770e+00],
   [ 1.40215334e+00],
   [ 4.00371949e-01]],

  [[-4.60451478e-02],
   [-8.77762996e-01],
   [-3.01140799e-01]],

  [[-1.23105357e-01],
   [-6.19886016e-02],
   [-3.30370451e-01]],

  [[-1.89536544e-01],
   [ 1.35698015e+00],
   [ 7.67071679e-01]],

  [[-3.52933601e-03],
   [-8.45876069e-01],
   [-1.15411647e+00]],

  [[-3.54285710e-01],
   [-6.06723989e-01],
   [ 2.80795850e-01]],

  [[-8.84388564e-03],
   [-1.97508154e-01],
   [ 9.47887592e-02]],

  [[ 4.69460314e-01],
   [ 1.09657015e+00],
   [ 9.90280325e-01]],

  [[-2.47873394e-02],
   [-8.99020960e-01],
   [ 3.49884215e-01]],

  [[ 1.10985623e+00],
   [ 5.43863228e-01],
   [ 1.16046763e-01]],

  [[ 4.96155750e-02],
   [ 1.60410388e+00],
   [ 4.53516860e-01]],

  [[-3.38342256e-01],
   [-1.02656873e+00],
   [-8.77762996e-01]],

  [[ 1.46326997e+00],
   [ 3.92400222e-01],
   [-2.71911127e-01]],

  [[ 1.24018490e-01],
   [-8.03360179e-01],
   [ 1.23740405e+00]],

  [[ 8.04273234e-01],
   [-8.59039774e-02],
   [-9.14964414e-01]],

  [[-7.58186995e-01],
   [ 5.27919579e-01],
   [-4.36660293e-01]],

  [[-6.06723989e-01],
   [-9.12185270e-02],
   [ 9.23849332e-01]],

  [[ 1.50047143e+00],
   [ 5.51834955e-01],
   [-2.18766196e-01]],

  [[-6.59868900e-01],
   [-7.63501408e-01],
   [ 1.55095917e+00]],

  [[ 8.68170323e-02],
   [-3.38342256e-01],
   [-6.19886016e-02]],

  [[ 1.65724879e+00],
   [ 7.83015230e-01],
   [-4.60451478e-02]],

  [[-7.71473154e-01],
   [-5.56236313e-01],
   [ 6.39523951e-01]],

  [[ 5.27919579e-01],
   [-2.55967615e-01],
   [-3.52933601e-03]],

  [[-1.07161864e-01],
   [ 6.02322494e-01],
   [ 5.73092959e-01]],

  [[ 2.83575716e-02],
   [-8.40561578e-01],
   [-4.73861711e-01]],

  [[ 1.42606852e+00],
   [ 9.69022321e-01],
   [-1.60306775e-01]],

  [[-4.65889965e-01],
   [-3.09766509e-01],
   [ 5.76772628e-01]],

  [[ 5.45646319e-02],
   [-1.84274644e-01],
   [-1.15456571e-01]],

  [[ 7.58938159e-01],
   [ 1.83169101e+00],
   [ 5.56532020e-01]],

  [[-1.42534710e-02],
   [-4.35258347e-01],
   [-2.40948400e-01]],

  [[ 2.81259578e-01],
   [ 4.63425140e-01],
   [-4.66384676e-02]],

  [[-1.84274644e-01],
   [ 6.13205764e-01],
   [ 2.28508075e+00]],

  [[ 2.04345226e-01],
   [-3.34055253e-01],
   [-6.98386391e-01]],

  [[-2.40948400e-01],
   [-5.24317072e-01],
   [-2.53092776e-01]],

  [[-4.47402717e-01],
   [-6.41712664e-01],
   [-2.10908245e-03]],

  [[ 3.94607007e-01],
   [ 1.73858431e+00],
   [-9.37225696e-01]],

  [[-9.25081337e-01],
   [ 4.31040143e-01],
   [ 1.47671512e-01]],

  [[-4.79787710e-01],
   [-1.59985926e-01],
   [-1.39745288e-01]],

  [[ 8.69496285e-02],
   [ 1.26090561e+00],
   [ 1.35401237e+00]],

  [[-3.44940791e-02],
   [-9.52159626e-02],
   [-3.66440241e-01]],

  [[-4.67643337e-01],
   [ 6.45590731e-01],
   [ 9.90940171e-02]],

  [[-1.03312182e-01],
   [-3.78584612e-01],
   [ 7.83226966e-01]],

  [[ 2.42271702e+00],
   [ 4.24202730e-02],
   [-2.16659658e-01]],

  [[-1.03312182e-01],
   [ 6.01061346e-01],
   [-2.28804020e-01]],

  [[-5.16220826e-01],
   [-4.23113976e-01],
   [-8.30715442e-02]],

  [[ 1.24066500e+00],
   [ 8.27756203e-01],
   [-3.09766509e-01]],

  [[-5.04076454e-01],
   [ 2.61018941e-01],
   [ 2.69115190e-01]],

  [[ 2.36730193e-01],
   [-8.30715442e-02],
   [-2.28804020e-01]],

  [[ 1.37425304e+00],
   [ 2.24585864e-01],
   [-2.10908245e-03]],

  [[-1.72130285e-01],
   [-3.21910879e-01],
   [ 3.37933293e-01]],

  [[ 1.80056479e-01],
   [-7.09271854e-02],
   [-1.15456571e-01]],

  [[-8.30715442e-02],
   [ 1.80740235e+00],
   [ 1.17994309e+00]],

  [[ 9.90940171e-02],
   [ 4.24202730e-02],
   [ 4.75569499e-01]]]]

    print('len:', len(X_test[0]))   
    predict_x=model.predict(X_test) 
    y_pred=np.argmax(predict_x,axis=1)
    print('pred:', y_pred)
    assert(y_pred == 4)

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

def write_res_into_file_converter(file_name, res_list):
    with open(file_name, 'w') as f:
        for v in res_list:
            f.write(str(v))
            f.write('\n')
        f.close()

    print('write_res_into_file, len:', len(res_list))
    
    return True

def run():
    # execute this when you want to load the model
    from keras.models import load_model
    MODEL_SAVED_PATH = 'saved-model'

    ml = load_model(MODEL_SAVED_PATH)
    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3
    # Chores:4, Desk_activity:5, Dining_room_activity:6, Eve_Med:7, Leaving_home: 8, Meditate:9, Morning_Med:10, Reading: 11
    
    # !!!Notice: the list of class_names should correspond to the list of the result of 'ls ./labelled', like '0  1  10  11  2  3  4  5  6  7  8  9'
    class_names=['bathroom','bedroom', 'morning_med', 'reading', 'kitchen','livingroom', 'chores', 'desk_activity', 'dining_room_activity',
                 'eve_med', 'leaving_home', 'meditate']

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
        
        write_res_into_file(ASCC_DATA_RES_FILE, res)            


MOTION_FOLDER_TEST = '/home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES/room_motion_activity/motion/test/'
MOTION_FOLDER = '/home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES/room_motion_activity/motion/'

MOTION_TXT = 'motion.txt'

TARGET_FILE = './ascc_v1_test_raw.txt'

def convert(act, time, motion_file, target, user='ascc'):
    t_str_list = []
    cur_time = time
    with open(motion_file, 'r') as f:
        for index, line in enumerate(f):
            # print("Line {}: {}".format(index, line.strip()))

            s_str = str(line.strip())
            xyz_arr = s_str.split('\t')

            x = xyz_arr[0]
            y = xyz_arr[1]
            z = xyz_arr[2]

            cur_time = cur_time + 1

            t_str = str(user) + ',' + str(act) + ',' + str(cur_time) + ',' + str(x) + ',' + str(y) + ',' + str(z)

            t_str_list.append(t_str)

        f.close()
    print('act:', act)
    print('time:', cur_time)
    print('motion_file:', motion_file)
    print('target:', target)
    print('user:', user)
    print('len(t_str_list:', len(t_str_list))

    write_res_into_file_converter(target, t_str_list)

    return len(t_str_list)


"""file_dict_test = {
    'Sitting': ['20220729102338'],
    'Standing': ['20220729103312'],
    'Walking': ['20220729105500'],
    'Jogging': ['20220729111901'],
    'Laying': ['20220729104633'],
    'Squating': ['20220729110944']
}"""


def sorter_take_count(elem):
    # print('elem:', elem)
    return elem[1]

def get_frames(df, frame_size, hop_size):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]

        frames_xyz = []
        for tmp_i in range(frame_size):
            frames_xyz.append([x[tmp_i], y[tmp_i], z[tmp_i]])
        frames.append(frames_xyz)

        # frames.append([x, y, z])
        # print('x:',len(x))
        # print('frames:', frames)
        # frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
        # print('reshape=======================')
        # print(frames)
        # exit(0)
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    print('frame shape:', frames.shape)
    labels = np.asarray(labels)

    return frames, labels


def get_data_from_motion_file(motion_file):

    # DATA_SET_FILE = 'ascc_dataset/ascc_v1_raw.txt'
    DATA_SET_FILE = motion_file
    pd.read_csv(DATA_SET_FILE)
    file = open(DATA_SET_FILE)

    lines = file.readlines()

    processedList = []

    for i, line in enumerate(lines):
        try:
            line = line.split(',')
            last = line[5].split(';')[0]
            last = last.strip()
            if last == '':
                break;
            temp = [line[0], line[1], line[2], line[3], line[4], last]
            processedList.append(temp)
        except:
            print('Error at line number: ', i)


    print('len processedList:', len(processedList))

    columns = ['user', 'activity', 'time', 'x', 'y', 'z']
    data = pd.DataFrame(data = processedList, columns = columns)
    # data.head()

    # data.shape

    # data.info()


    data.isnull().sum()

    data['activity'].value_counts()

    data['x'] = data['x'].astype('float')
    data['y'] = data['y'].astype('float')
    data['z'] = data['z'].astype('float')


    data.info()

    # sample rate
    Fs = 95

    activities = data['activity'].value_counts().index
    print('activities:', activities)


    df = data.drop(['user', 'time'], axis = 1).copy()
    print('df.head:', df.head())

    print('counts:', df['activity'].value_counts())
    sd = sorted(df['activity'].value_counts().items(), key=sorter_take_count, reverse=False)
    print('sd:', sd)



    min_count = sd[0][1] #15606 # ASCC dataset
    # min_count = 3555 # for WISDM dataset

    Walking = df[df['activity']=='Walking'].head(min_count).copy()
    Jogging = df[df['activity']=='Jogging'].head(min_count).copy()
    Laying = df[df['activity']=='Laying'].head(min_count).copy()
    Squating = df[df['activity']=='Squating'].head(min_count).copy()
    Sitting = df[df['activity']=='Sitting'].head(min_count).copy()
    Standing = df[df['activity']=='Standing'].copy()

    balanced_data = pd.DataFrame()
    balanced_data = balanced_data.append([Walking, Jogging, Laying, Squating, Sitting, Standing])
    # balanced_data.shape

    balanced_data['activity'].value_counts()

    print('balanced_data:', balanced_data)
    print(balanced_data.head())

    from sklearn.preprocessing import LabelEncoder

    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['activity'])
    print('head:',balanced_data.head())
    print('================ label mapping')
    print(balanced_data.values.tolist())

    print('label:',label.classes_)
    X = balanced_data[['x', 'y', 'z']]
    y = balanced_data['label']
    #print('label y:', y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
    scaled_X['label'] = y.values

    # scaled_X

    import scipy.stats as stats

    frame_size = Fs*3 # 80
    hop_size = Fs*2 # 40



    X, y = get_frames(scaled_X, frame_size, hop_size)
    print('X:', X.shape)
    print('y:', y.shape)

    # X.shape: (520, 285, 3)  
    # y.shape: (520,)

    return X, y



def get_activity_by_motion_dnn(time_str, action='moiton'):


    date_format_str = '%Y-%m-%d %H:%M:%S'
    print("get_activity time_str:", time_str)
    d_act = time_str

    # image_dir_name = get_exist_image_dir(time_str, action)
    motion_file = MOTION_FOLDER_TEST + d_act + '/' + MOTION_TXT

    act = 'Sitting'
    time = time_str  # 12585782270000
    motion_file = motion_file
    target = TARGET_FILE
    user = 'ascc'

    c_len = convert(act, int(time), motion_file, target, user)
    print('convert len:', c_len)

    X, y = get_data_from_motion_file(target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


    # X_train.shape, X_test.shape
    print('X_train.shape:', X_train.shape)
    print('X_test.shape:', X_test.shape)
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    X_test = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    # execute this when you want to load the model
    from keras.models import load_model
    MODEL_SAVED_PATH = 'motion-saved-model'

    model = load_model(MODEL_SAVED_PATH)

    model.summary()

    predict_x=model.predict(X_test)
    y_pred=np.argmax(predict_x,axis=1)

    print('y_pred len:', len(y_pred))
    print('y_pred:', y_pred)


    exit(0)



    t_data_list = []
    t_arr = []

    Fs = 95
    frame_size = Fs*3 # 80
    hop_size = Fs*2 # 40

    # x_train shape: (300, 3, 1)
    shape_len = frame_size
    #shape_len = 80

    N_FEATURES = 3
    # frame_size = frame_size


    columns = ['x', 'y', 'z']
    # data = pd.DataFrame(data = processedList, columns = columns)
    # data.head()

    # data.shape

    # data.info()


    # data.isnull().sum()

    # data['activity'].value_counts()

    # data['x'] = data['x'].astype('float')
    # data['y'] = data['y'].astype('float')
    # data['z'] = data['z'].astype('float')

    processedList = []

    with open(motion_file, 'r') as f:
        for index, line in enumerate(f):
            # print("Line {}: {}".format(index, line.strip()))

            s_str = str(line.strip())
            xyz_arr = s_str.split('\t')

            x = float(xyz_arr[0])
            y = float(xyz_arr[1])
            z = float(xyz_arr[2])

            

            t_arr = []

            t_arr.append(x)
            t_arr.append(y)
            t_arr.append(z)

            t_data_list.append(t_arr)

            # processedList = t_data_list
            # data = pd.DataFrame(data = processedList, columns = columns)
            # data['x'] = data['x'].astype('float')
            # data['y'] = data['y'].astype('float')
            # data['z'] = data['z'].astype('float')

            # print(data)


        f.close()

    print('len t_data_list:', len(t_data_list))

    n = shape_len
    chunks = [t_data_list[i:i+n] for i in range(0, len(t_data_list), n)]

    # execute this when you want to load the model
    from keras.models import load_model
    MODEL_SAVED_PATH = 'motion-saved-model'

    model = load_model(MODEL_SAVED_PATH)


    for i in range(len(chunks)):
        if len(chunks[i]) != shape_len:
            continue

        frames = []
        frames_xyz = chunks[i]
        frames.append(frames_xyz)

        frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)

        X_test = []
        X_test.append(frames)

        X_test = np.asarray(X_test).reshape(-1, frame_size, N_FEATURES)
        print('X_test shape:', X_test.shape)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        

        print('i:',i )
        # print('len:', len(X_test[0]))
        print('X_test:', X_test)
        predict_x=model.predict(X_test)
        y_pred=np.argmax(predict_x,axis=1)
        print('pred:', y_pred)
        # assert(y_pred == 4)


'''
    'Sitting': ['20220729102338'],
    'Standing': ['20220729103312'],
    'Walking': ['20220729105500'],
    'Jogging': ['20220729111901'],
    'Laying': ['20220729104633'],
    'Squating': ['20220729110944']

    walking: 5
stand: 4
Squating:3
Sitting:2
Laying:1
Jogging:0
'''
if __name__ == "__main__":
    print('Test running:===========================================================\n')

    log.init_log("./log/my_program")  # ./log/my_program.log./log/my_program.log.wf7
    logging.info("Hello World!!!")
    #test()
    # run()
    
    # get_activity_by_motion_dnn('20220729103103', 'test' ) # 5 wrong

    # get_activity_by_motion_dnn('20220729102338', 'Sitting' ) # 4, 0 wrong
    # get_activity_by_motion_dnn('20220729103312', 'Stand' ) # 4, g
    # get_activity_by_motion_dnn('20220729105500', 'walking') # 0, w
    # get_activity_by_motion_dnn('20220729111901', 'jogging') # 0, g
    get_activity_by_motion_dnn('20220729104633', 'Laying') # 0,4, wrong
    # get_activity_by_motion_dnn('20220729110944', 'Squating') # 0.4 wrong




