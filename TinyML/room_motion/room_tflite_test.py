from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import time
import os
# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img



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
    model_path = "./home_default_model.tflite"

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
    #image = Image.open(data_folder + "test.jpg").convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "bathroom2.jpg").convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "bedroom3.jpg").convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "kitchen_test.jpg").convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "livingroom_test.jpg").convert('RGB').resize((width, height))

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


def get_confusion_matrix():
    dir = './room_testset/'  # ./tf_testset
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
    mat = confusion_matrix(y_test, y_pred)
    cm = plot_confusion_matrix(conf_mat=mat, class_names=class_names, show_normed=True, figsize=(7,7))
    plt.show()
    plt.savefig("room_cm.png")


# makes the prediction of the file path image passed as parameter 
def predict(file, model, to_class, width, height):
    # im = load_img(file, target_size=(width, height))
    # x = img_to_array(im)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # tmp  = model.predict(x)
    # #print('tmp:', tmp)
    # index = model.predict(x).argmax()
    # return to_class[index]

    # data_folder_image = "./room_samples_0831/"

    # #model_path = "./home_model.tflite"
    # model_path = "./home_default_model.tflite"

    # # bathroom: 0, bedroom:1, kitchen:2, livingroom:3, hallway:4, door:5
    # labels=['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']
    # # label_path = data_folder + "labels_home_v1.txt"
    # print("labels:", labels)

    # interpreter = Interpreter(model_path)
    # print("Model Loaded Successfully.")

    # interpreter.allocate_tensors()
    # _, height, width, _ = interpreter.get_input_details()[0]['shape']
    # print("Image Shape (", width, ",", height, ")")

    # Load an image to be classified.
    #image = Image.open(data_folder + "test.jpg").convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "bathroom2.jpg").convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "bedroom3.jpg").convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "kitchen_test.jpg").convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "livingroom_test.jpg").convert('RGB').resize((width, height))

    # image = Image.open(data_folder_image + "Bedroom.jpg").convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "Livingroom(WatchTV).jpg").convert('RGB').resize((width, height))

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


test()
# get_confusion_matrix()
