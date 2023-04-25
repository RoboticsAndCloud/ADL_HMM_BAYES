import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


random_seed = 42   
n_time_steps = 90
n_features = 3 
n_classes = 4
n_epochs = 300       
batch_size = 1024   
learning_rate = 0.0025
l2_loss = 0.0015
segments = []
labels = []



def load_data(act_index, motion_file):
    global segments
    global labels
    #print('act_index:', act_index)
    #print('motion_file:', motion_file)
    cnt = 0
    xs = []
    ys = []
    zs = []
    with open(motion_file, 'r') as f:
        for index, line in enumerate(f):
            # print("Line {}: {}".format(index, line.strip()))

            s_str = str(line.strip())
            xyz_arr = s_str.split('\t')
            
            x = xyz_arr[0]
            y = xyz_arr[1]
            z = xyz_arr[2]

            xs.append(x)
            ys.append(y)
            zs.append(z)

            cnt += 1

            if cnt >= n_time_steps:
              break
    f.close()

    if cnt < n_time_steps:
        return

    segments.append([xs, ys, zs])
    labels.append(act_index)
    




    return cnt



MOTION_FILE_PATH = '/home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_motion_activity/watch_data/Motion/'
class_labels = ['Standing', 'Sitting', 'Laying', 'Walking']
for i in range(0,  4):
    files = os.listdir(MOTION_FILE_PATH + '/' + class_labels[i])
    #print('files:', files
    sample_cnt = 0
    for f in files:
        target_f = MOTION_FILE_PATH + '/' + class_labels[i] + '/' + f
        #print('target_f:', target_f)
        if os.path.isfile(target_f):
            load_data(i, target_f)
            sample_cnt += 1

    print("act:", class_labels[i], ' sample_cnt:', sample_cnt)


print('len segments:', len(segments))
# segments = segments[:74970]
# labels = labels[:74970]
print('len segments:', len(segments))

reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)
print('len reshaped segments:', len(reshaped_segments))

labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

#exit(0)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size = 0.2, random_state = random_seed)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout
model = Sequential()
# RNN layer
model.add(LSTM(units = 128, input_shape = (X_train.shape[1], X_train.shape[2])))
# Dropout layer
model.add(Dropout(0.5)) 
# Dense layer with ReLu
model.add(Dense(units = 64, activation='relu'))
# Softmax layer
model.add(Dense(y_train.shape[1], activation = 'softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs = n_epochs, validation_split = 0.20, batch_size = batch_size, verbose = 1)


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

plot_learningCurve(history, n_epochs)



plt.plot(np.array(history.history['loss']), "r--", label = "Train loss")
plt.plot(np.array(history.history['accuracy']), "g--", label = "Train accuracy")
plt.plot(np.array(history.history['val_loss']), "r-", label = "Validation loss")
plt.plot(np.array(history.history['val_accuracy']), "g-", label = "Validation accuracy")
plt.title("Training session's progress over iterations")
plt.legend(loc='lower left')
plt.ylabel('Training Progress (Loss/Accuracy)')
plt.xlabel('Training Epoch')
plt.ylim(0) 
plt.show()
plt.savefig("motion_lstm.png")


loss, accuracy = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
print("Test Accuracy :", accuracy)
print("Test Loss :", loss)



# confusion matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# y_pred = model.predict_classes(X_test)

# print('X_test:', X_test)

predict_x = model.predict(X_test)
y_pred = np.argmax(predict_x, axis=1)
max_test = np.argmax(y_test, axis=1)

#print('y_pred len:', len(y_pred))
#print('y_pred:', y_pred)


plt.figure()
mat = confusion_matrix(max_test, y_pred)
cm = plot_confusion_matrix(conf_mat=mat, class_names=class_labels, show_normed=True, figsize=(7,7))
plt.show()
plt.savefig("watch_cm.png")

MODEL_SAVED_PATH = 'motion-lstm-saved-model'
# model.save_weights('model.h5')
model.save(MODEL_SAVED_PATH)


'''
predictions = model.predict(X_test)
class_labels = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
max_test = np.argmax(y_test, axis=1)
max_predictions = np.argmax(predictions, axis=1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)
sns.heatmap(confusion_matrix, xticklabels = class_labels, yticklabels = class_labels, annot = True, linewidths = 0.1, fmt='d', cmap = 'YlGnBu')
plt.title("Confusion matrix", fontsize = 15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

'''
