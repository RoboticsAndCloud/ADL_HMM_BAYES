# reference: https://www.tensorflow.org/model_optimization/guide/quantization/training_example

from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import time
import os
# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.layers import Conv2D, MaxPool2D
# from tensorflow.keras.optimizers import Adam
# print(tf.__version__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import seaborn as sns


MOTION_TFLITE_MODEL = './motion_default_model.tflite'

def sorter_take_count(elem):
    # print('elem:', elem)
    return elem[1]

# pd.read_csv('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
# file = open('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')

# DATA_SET_FILE = 'WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
DATA_SET_FILE = './ascc_v1_raw.txt'
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

data.isnull().sum()

data['activity'].value_counts()

data['x'] = data['x'].astype('float')
data['y'] = data['y'].astype('float')
data['z'] = data['z'].astype('float')


# data.info()

# sample rate
Fs = 90

activities = data['activity'].value_counts().index
# print('activities:', activities)

def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 7), sharex=True)
    plot_axis(ax0, data['time'], data['x'], 'X-Axis')
    plot_axis(ax1, data['time'], data['y'], 'Y-Axis')
    plot_axis(ax2, data['time'], data['z'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()
    plt.savefig(activity + ".png")

    

def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

for activity in activities:
    data_for_plot = data[(data['activity'] == activity)][Fs*10: Fs*18]
    print('data_for_plot', data_for_plot)
    plot_activity(activity, data_for_plot)

#exit(0)

df = data.drop(['user', 'time'], axis = 1).copy()
# print('df.head:', df.head())

# print('counts:', df['activity'].value_counts())
sd = sorted(df['activity'].value_counts().items(), key=sorter_take_count, reverse=False)
# print('sd:', sd)



min_count = sd[0][1] #15606 # ASCC dataset
# min_count = 3555 # for WISDM dataset

Walking = df[df['activity']=='Walking'].head(min_count).copy()
Jogging = df[df['activity']=='Jogging'].head(min_count).copy()
Laying = df[df['activity']=='Laying'].head(min_count).copy()
Jumping = df[df['activity']=='Jumping'].head(min_count).copy()
#Squating = df[df['activity']=='Squating'].head(min_count).copy()
Sitting = df[df['activity']=='Sitting'].head(min_count).copy()
Standing = df[df['activity']=='Standing'].copy()

balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([Walking, Jogging, Laying, Jumping, Sitting, Standing])
# balanced_data.shape

balanced_data['activity'].value_counts()

# print('balanced_data:', balanced_data)
# print(balanced_data.head())

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['activity'])
# print('head:',balanced_data.head())
# print('================ label mapping')
# print(balanced_data.values.tolist())

print('label:',label.classes_)
X = balanced_data[['x', 'y', 'z']]
y = balanced_data['label']
#print('label y:', y)

scaler = StandardScaler()
X = scaler.fit_transform(X)
print('scaler: mean, var', scaler.mean_, ' ', scaler.var_)

scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
scaled_X['label'] = y.values

# scaled_X

import scipy.stats as stats

frame_size = Fs*2 # 80
hop_size = Fs*2 # 40

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
    # print('frame shape:', frames.shape)
    labels = np.asarray(labels)

    return frames, labels

X, y = get_frames(scaled_X, frame_size, hop_size)

# print('X.shape:', X.shape)
# print('y.shape:', y.shape)
# exit(0)

# X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


# X_train.shape, X_test.shape
# print('X_train.shape:', X_train.shape)
# print('X_test.shape:', X_test.shape)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print('x_train shape:', X_train[0].shape)
print('X_test shape:', X_test[0].shape)
print('X_test len:', len(X_test))
print('y_test len:', len(y_test))

# tmp_xtest = X_test[0]
# tmp_ytest = y_test[0]
# print('tmp_xtest:', tmp_xtest)
# print('tmp_ytest:', tmp_ytest)


LABELS = label.classes_

def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=2):
  set_input_tensor(interpreter, image)

  time1 = time.time()
  interpreter.invoke()
  time2 = time.time()
  print("time2:", time2)
  classification_time = np.round(time2-time1, 3)
  print("invoken Time =", classification_time, "seconds.")

  output_details = interpreter.get_output_details()
  # print("output details:", output_details)
  output_details = interpreter.get_output_details()[0]
  # print("output2 details:", output_details)
  output = np.squeeze(interpreter.get_tensor(output_details['index']))
  print('output1:', output)
  prediction_classes = np.argmax(output)
  print('prediction_classes:', prediction_classes)

  scale, zero_point = output_details['quantization']
  # output = scale * (output - zero_point)
  # print('output:', output)

  # ordered = np.argpartition(-output, 1)
  # print('ordered:', ordered)

  return prediction_classes, output[prediction_classes]

def test():
    data_folder = "./"
    data_folder_image = "./room_samples_0831/"

    #model_path = "./home_model.tflite"
    model_path = "./motion_default_model.tflite"

    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3, hallway:4, door:5
    labels=['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']
    # label_path = data_folder + "labels_home_v1.txt"
    print("labels:", labels)

    interpreter = Interpreter(model_path)
    print("Model Loaded Successfully.")
    # Analyze the tflite model
    #tf.lite.experimental.Analyzer.analyze(model_content=interpreter)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print("input_details:", input_details)
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print("Image Shape (", width, ",", height, ")")

    # Load an image to be classified.

    image = Image.open(data_folder_image + "Bedroom.jpg").convert('RGB').resize((width, height))
    image = Image.open(data_folder_image + "Kitchen.jpg").convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "Livingroom(WatchTV).jpg").convert('RGB').resize((width, height))


    # Classify the image.
    time1 = time.time()
    print("time1:", time1)
    label_id, prob = classify_image(interpreter, image)
    print("label id:", label_id)
    time2 = time.time()
    print("time2:", time2)
    classification_time = np.round(time2-time1, 3)
    print("Classificaiton Time =", classification_time, "seconds.")

    # Read class labels.
    # labels = load_labels(label_path)

    # Return the classification label of the image.
    classification_label = labels[label_id]
    print("Image Label is :", classification_label, ", with Accuracy :", np.round(prob*100, 2), "%.")


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


def predict_tflite(test_motion_data):


    model_path = MOTION_TFLITE_MODEL

    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3, hallway:4, door:5
    # labels=['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']
    # # label_path = data_folder + "labels_home_v1.txt"
    # print("labels:", labels)


    interpreter = Interpreter(model_path)
    print("Model Loaded Successfully.")

    interpreter.allocate_tensors()

    shape = interpreter.get_input_details()[0]['shape']
    print("interpreter input Shape:", shape)



    # set_input_tensor(interpreter, image)
    input_index = interpreter.get_input_details()[0]['index']

    
    # todo for loop to get the result
    test_motion = np.expand_dims(test_motion_data, axis=0).astype(np.float32)
    print("test_motion_shape:", test_motion.shape)


    interpreter.set_tensor(input_index, test_motion)



    time1 = time.time()
    interpreter.invoke()
    time2 = time.time()
    print("time2:", time2)
    classification_time = np.round(time2-time1, 3)
    print("invoken Time =", classification_time, "seconds.")

    output_details = interpreter.get_output_details()
    # print("output details:", output_details)
    output_details = interpreter.get_output_details()[0]
    # print("output2 details:", output_details)
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    print('output1:', output)
    prediction_classes = np.argmax(output)
    print('prediction_classes:', prediction_classes, " ", LABELS[prediction_classes])

    return prediction_classes

def get_confusion_matrix():
    dir = './room_testset/'  # ./tf_testset
    path = dir
    global y_test, X_test
    y_test = y_test
    y_pred = []

    print('X_test shape:', X_test.shape)
    class_names = LABELS


    for d in X_test:
        print('d shape:', d.shape)
        pre = predict_tflite(d)
        y_pred.append(pre)
    
    # for fn in os.listdir(path):
    #     if os.path.isdir(dir + '/' + fn):
    #         print(fn)
    #         test_dir = dir + '/' + fn
    #         sample_cnt = get_file_count_of_dir(test_dir)
    #         print('sample cnt:', sample_cnt)
            
    #         test_truth_label = []
    #         for i in range(sample_cnt):
    #             index = int(fn)
    #             print('test lable:', class_names[index])
    #             test_truth_label.append(class_names[index])
                

    #         y_test.extend(test_truth_label)
    #         print('len test_truth_label:', len(test_truth_label))
    #         print('len y_test:', len(y_test))

    #         predict_res = test_confusion_matrix(test_dir)
    #         print('len(predict_res):', len(predict_res))
    #         y_pred.extend(predict_res)


    import matplotlib.pyplot as plt
    # confusion matrix
    from mlxtend.plotting import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix

    print('y_pred len:', len(y_pred))
    print('y_pred:', y_pred)
    print('y_test len:', len(y_test))
    print('y_test:', y_test)

    plt.figure()
    mat = confusion_matrix(y_test, y_pred)
    cm = plot_confusion_matrix(conf_mat=mat, class_names=class_names, show_normed=True, figsize=(7,7))
    plt.show()
    plt.savefig("room_tflite_cm.png")


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
    return to_class[label_id]


def test_confusion_matrix(file_dir):
        #model_path = "./home_model.tflite"
    model_path = "./home_default_model.tflite"

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
            file = img
            label = predict(file, interpreter, class_names, width, height)
            predict_res.append(label)

    return predict_res


MOTION_FOLDER_TEST = '/home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES/room_motion_activity/motion/test/'
MOTION_FOLDER_0802 = '/home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES/room_motion_activity/motion_0802/'
MOTION_FOLDER = '/home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES/room_motion_activity/motion/'

MOTION_TXT = 'motion.txt'

TARGET_FILE = './ascc_v1_test_raw.txt'

def write_res_into_file_converter(file_name, res_list):
    with open(file_name, 'w') as f:
        for v in res_list:
            f.write(str(v))
            f.write('\n')
        f.close()

    print('write_res_into_file, len:', len(res_list))
    
    return True

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


def get_data_scaler():

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


    # print('len processedList:', len(processedList))

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


    # data.info()


    activities = data['activity'].value_counts().index
    # print('activities:', activities)


    df = data.drop(['user', 'time'], axis = 1).copy()
    # print('df.head:', df.head())

    # print('counts:', df['activity'].value_counts())
    sd = sorted(df['activity'].value_counts().items(), key=sorter_take_count, reverse=False)
    # print('sd:', sd)



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

    # balanced_data['activity'].value_counts()

    # print('balanced_data:', balanced_data)
    # print(balanced_data.head())

    from sklearn.preprocessing import LabelEncoder

    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['activity'])
    # print('head:',balanced_data.head())
    # print('================ label mapping')
    #print(balanced_data.values.tolist())

    # print('label:',label.classes_)
    X = balanced_data[['x', 'y', 'z']]
    y = balanced_data['label']
    #print('X before:', X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #print('X after:', X)

    return scaler


def get_data_from_motion_file(motion_file):

    pd.read_csv(motion_file)
    file = open(motion_file)

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


    # print('len processedList:', len(processedList))

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


    # data.info()



    activities = data['activity'].value_counts().index
    # print('activities:', activities)


    df = data.drop(['user', 'time'], axis = 1).copy()
    # print('df.head:', df.head())

    # print('counts:', df['activity'].value_counts())
    sd = sorted(df['activity'].value_counts().items(), key=sorter_take_count, reverse=False)
    # print('sd:', sd)



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

    # print('balanced_data:', balanced_data)
    # print(balanced_data.head())

    from sklearn.preprocessing import LabelEncoder

    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['activity'])
    # print('head:',balanced_data.head())
    # print('================ label mapping')
    #print(balanced_data.values.tolist())

    # print('label:',label.classes_)
    X = balanced_data[['x', 'y', 'z']]
    y = balanced_data['label']

    #scaler = StandardScaler()
    scaler = get_data_scaler()
    # print('scaler: mean, var', scaler.mean_, ' ', scaler.var_)
    X = scaler.transform(X)


    scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
    scaled_X['label'] = y.values

    # scaled_X

    import scipy.stats as stats

    frame_size = Fs*2 # 80
    hop_size = Fs*2 # 40



    X, y = get_frames(scaled_X, frame_size, hop_size)
    # print('X:', X.shape)
    # print('y:', y.shape)

    # X.shape: (520, 285, 3)  
    # y.shape: (520,)

    return X, y



def get_activity_by_motion_dnn(time_str, action='moiton'):

    print("get_activity time_str:", time_str)
    d_act = time_str
        # image_dir_name = get_exist_image_dir(time_str, action)
    motion_file = MOTION_FOLDER_TEST + d_act + '/' + MOTION_TXT
    get_activity_prediction(motion_file)

def get_activity_prediction(motion_file, act = 'Sitting', time_str = '0'):

    act = act
    # time = time_str  # 12585782270000
    motion_file = motion_file
    target = TARGET_FILE
    user = 'ascc'

    c_len = convert(act, int(time_str), motion_file, target, user)
    # print('convert len:', c_len)

    X, y = get_data_from_motion_file(target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 0, stratify = y)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    # print('x_train shape:', X_train[0].shape)
    # print('X_test len:', len(X_test))
    # tmp_xtest = X_test[0]
    # tmp_ytest = y_test[0]
    #print('tmp_xtest:', tmp_xtest)
    #print('tmp_ytest:', tmp_ytest)
    


    # X_train.shape, X_test.shape
    # print('X_train.shape:', X_train.shape)
    # print('X_test.shape:', X_test.shape)
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    #X_test = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    # execute this when you want to load the model
    # from keras.models import load_model
    # MODEL_SAVED_PATH = 'motion-saved-model'

    # model = load_model(MODEL_SAVED_PATH)

    # model.summary()

    # print('====X_test :', X_test)
    print('X_test shape:', X_test.shape)

    # exit(0)

    model_path = MOTION_TFLITE_MODEL

    # bathroom: 0, bedroom:1, kitchen:2, livingroom:3, hallway:4, door:5
    # labels=['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']
    # # label_path = data_folder + "labels_home_v1.txt"
    # print("labels:", labels)


    interpreter = Interpreter(model_path)
    print("Model Loaded Successfully.")

    interpreter.allocate_tensors()

    shape = interpreter.get_input_details()[0]['shape']
    print("interpreter input Shape:", shape)



    # set_input_tensor(interpreter, image)
    input_index = interpreter.get_input_details()[0]['index']
    test_motion = np.float32(X_test)
    print("test_motion_shape:", test_motion.shape)
    
    # todo for loop to get the result
    test_motion = np.expand_dims(test_motion[0], axis=0).astype(np.float32)

    interpreter.set_tensor(input_index, test_motion)



    time1 = time.time()
    interpreter.invoke()
    time2 = time.time()
    print("time2:", time2)
    classification_time = np.round(time2-time1, 3)
    print("invoken Time =", classification_time, "seconds.")

    output_details = interpreter.get_output_details()
    # print("output details:", output_details)
    output_details = interpreter.get_output_details()[0]
    # print("output2 details:", output_details)
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    print('output1:', output)
    prediction_classes = np.argmax(output)
    print('prediction_classes:', prediction_classes, " ", LABELS[prediction_classes])


    # predict_x=model.predict(X_test)
    # y_pred=np.argmax(predict_x,axis=1)

    prob =  []

    # for i in range(len(predict_x)):
    #     prob.append(predict_x[i][y_pred[i]])
    
    # print('y_pred len:', len(y_pred), 'len(prob):', len(prob))
    # print('y_pred:', y_pred)
    # print('prob:', prob)

    return 0

# get_activity_by_motion_dnn('20220802154633', 'Sitting' ) # 4, w
# get_activity_by_motion_dnn('20220802155756', 'Stand' ) # 4, g
# get_activity_by_motion_dnn('20220802163654', 'walking') # 5, g
# get_activity_by_motion_dnn('20220802164744', 'jogging') # 0, g
# get_activity_by_motion_dnn('20220802161356', 'Laying') # 1, g

print("labels:")
print(LABELS)
# test()
get_confusion_matrix()
