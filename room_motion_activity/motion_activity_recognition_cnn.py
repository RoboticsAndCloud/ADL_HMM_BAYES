"""
@Brief: Human activity recognition based on motion data(accelerometer)
@Author: Frank
@Date: 07/29/2022
1) Github:https://github.com/RoboticsAndCloud/Human-Activity-Recognition-Using-Accelerometer-Data-and-CNN
2) Dataset Link: http://www.cis.fordham.edu/wisdm/dataset.php
    The WISDM dataset contains six different labels (Downstairs, Jogging, Sitting, Standing, Upstairs, Walking). 
3) ASCC DATASET
    The WISDM dataset contains six different labels (Squating, Jogging, Sitting, Standing, Laying, Walking). 
    
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.__version__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import seaborn as sns



def sorter_take_count(elem):
    # print('elem:', elem)
    return elem[1]

# pd.read_csv('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
# file = open('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')

# DATA_SET_FILE = 'WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
DATA_SET_FILE = 'ascc_dataset/ascc_v1_raw.txt'
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
    data_for_plot = data[(data['activity'] == activity)][:Fs*10]
    print('data_for_plot', data_for_plot)
    plot_activity(activity, data_for_plot)



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

X, y = get_frames(scaled_X, frame_size, hop_size)

# X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


# X_train.shape, X_test.shape
print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print('x_train shape:', X_train[0].shape)
print('X_test shape:', X_test[0].shape)
print('X_test len:', len(X_test))
tmp_xtest = X_test[0]
tmp_ytest = y_test[0]
print('tmp_xtest:', tmp_xtest)
print('tmp_ytest:', tmp_ytest)
# exit(0)

# CNN model
model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))

model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

epochs = 20

history = model.fit(X_train, y_train, epochs = epochs, validation_data= (X_test, y_test), verbose=1)

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

plot_learningCurve(history, epochs)

# confusion matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# y_pred = model.predict_classes(X_test)

# print('X_test:', X_test)

predict_x=model.predict(X_test) 
y_pred=np.argmax(predict_x,axis=1)

plt.figure()
mat = confusion_matrix(y_test, y_pred)
cm = plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7,7))
plt.show()
plt.savefig("cm.png")

# # confusion matrix
# LABELS = label.classes_

# plt.figure(figsize=(6, 4))
# sns.heatmap(mat,
#             cmap='coolwarm',
#             linecolor='white',
#             linewidths=1,
#             xticklabels=LABELS,
#             yticklabels=LABELS,
#             annot=True,
#             fmt='d')
# plt.title('Confusion Matrix')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.show()
# plt.savefig("confusion_matrix.png")

MODEL_SAVED_PATH = 'motion-saved-model'
# model.save_weights('model.h5')
model.save(MODEL_SAVED_PATH)
















