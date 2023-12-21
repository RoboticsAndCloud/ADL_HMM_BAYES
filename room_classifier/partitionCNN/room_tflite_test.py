import imghdr
from tflite_runtime.interpreter import Interpreter
#from keras.models import load_model
from PIL import Image
import numpy as np
import time
import os
import socket
import pickle
# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
IMG_WIDTH, IMG_HEIGHT = 299, 299
model_path = "./watch-saved-model-alex_multioutput.tflite"

def AlexnetModel2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),num_classes=6, l2_reg=0.):
    """
    https://github.com/visionatseecs/keras-starter/blob/main/keras_alexnet.ipynb
    https://colab.research.google.com/gist/SuryanarayanaY/07c54cc27046514599476fc0edd9b58f/43379-sr1.ipynb#scrollTo=h4jjjeqNfGMc
    https://github.com/pravinkr/alexnet-cifar10-using-keras/blob/master/cifar_10_with_Alexnet.ipynb
    """
    print('AlexnetModel2:')
    inputs = Input(shape=input_shape)

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), padding='same',  activation='relu',name='conv_1')(inputs)
    #conv_1 = BatchNormalization()(conv_1)
    conv_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv_1)
    conv_1 = ZeroPadding2D((2,2))(conv_1)

    conv_2 = Conv2D(256,(5,5),activation='relu', padding='same')(conv_1)
    conv_2 = MaxPooling2D((3,3), strides=(2, 2), padding='same')(conv_2)
    conv_2 = ZeroPadding2D((1,1))(conv_2)

    conv_3 = Conv2D(384, (3, 3), activation='relu', padding='same', name='Part_1')(conv_2)
    conv_4 = ZeroPadding2D((1,1))(conv_3)

    conv_4 = Conv2D(384, (3,3),activation='relu', padding='same')(conv_3)
    conv_4 = ZeroPadding2D((1,1))(conv_4)

    conv_5 = Conv2D(384, (3,3),activation='relu', padding='same')(conv_4)
    conv_5 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv_5)

    dense_1 = Flatten()(conv_5)
    dense_1 = Dense(384, activation='relu')(dense_1)
    dense_1 = Dropout(0.5)(dense_1)

    dense_2 = Dense(384, activation='relu')(dense_1)
    dense_2 = Dropout(0.5)(dense_2)

    dense_3 = Dense(num_classes,activation='softmax', name='Part_Final')(dense_2)

    p_outputs = [conv_3, dense_3]
    alexnet = Model(inputs=inputs, outputs=dense_3)
    #alexnet = Model(inputs=inputs, outputs=p_outputs)
    alexnet.summary()

    learning_rate = 0.001
    batch_size = 128
    alexnet.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    return alexnet

MODEL_SAVED_PATH = 'watch-saved-model-alex'



def server_call(model, inputs, layer_start):
  for layer in model.layers[layer_start:]:
    inputs = layer(inputs)
  return inputs


def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=2):
  print('interpreter:', interpreter)
  set_input_tensor(interpreter, image)

  interpreter.invoke()

  layer_index = 1  # Replace with the index of the layer you are interested in
  #output_tensor = interpreter.tensor(interpreter.get_output_details()[layer_index]['index'])
  #print('layer {:10s}, output_tensor:{}'.format(layer_index, output_tensor))

 # output_details = interpreter.get_output_details()
 ## print("output details:", output_details)
 # output_details = interpreter.get_output_details()[0]
 ## print("output2 details:", output_details)
 # p_inputs = output_details
 # print('part1 output :', interpreter.get_tensor(output_details['index']))
 # # send out to the server
 # output = np.squeeze(interpreter.get_tensor(output_details['index']))
 # print('output1:', output)
 # print('output1:', type(output))
 # print('output1:', output.shape)
 # prediction_classes = np.argmax(output)
 # print('prediction_classes:', prediction_classes)

 # scale, zero_point = output_details['quantization']
 # # output = scale * (output - zero_point)
 # # print('output:', output)

 # # ordered = np.argpartition(-output, 1)
 # # print('ordered:', ordered)
 # #layer_end = 8
 # #base_model = AlexnetModel2()
 # #base_model.summary()
 # #res = server_call(base_model, p_inputs, layer_end)
 # #print('res:', res)

 # # Send partition result to server

 # partition_outputs = interpreter.get_tensor(output_details['index'])
 # #saver_part=np.save('partial_data.npy', partition_outputs.numpy())
 # print('partition_outputs:')
 # print(partition_outputs)
 # part_output= partition_outputs
 # 
 # 
 # 
 # TCP_IP2 = '10.227.97.123'
 # TCP_PORT2 = 5006
 # 
 # s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 # s.connect((TCP_IP2,TCP_PORT2))
 # 
 # data_part=pickle.dumps(part_output, protocol=pickle.HIGHEST_PROTOCOL)
 # 
 # 
 # s.sendall(data_part)
 # 
 # #data=s.recv(4096)
 # #data_arr=pickle.loads(data)
 # 
 # #data=s.recv(4096)
 # s.close()
 # 
 # print('Final Output')

  output_details = interpreter.get_output_details()
  #print("output details:", output_details)
  output_details = interpreter.get_output_details()[1]
  #print("output2 details:", output_details)
  p_inputs = output_details
  #print('part1 output :', interpreter.get_tensor(output_details['index']))
  # send out to the server
  output = np.squeeze(interpreter.get_tensor(output_details['index']))
  #print('output1:', output)
  #print('output1:', type(output))
  #print('output1:', output.shape)
  prediction_classes = np.argmax(output)
  #print('prediction_classes:', prediction_classes)

  return prediction_classes, output[prediction_classes]

def test():
    data_folder = "./"
    data_folder_image = "./room_samples_0831/"

    #model_path = "./home_model.tflite"
    #model_path = "./home_model_default_onlinedataset.tflite"
    #model_path = "./watch-saved-model-alex_multioutput.tflite"
    global model_path

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
    test_img = './Images_test' + '/' + 'hunter_room.jpg'
    #test_img = './Images_test' + '/' + 'bedroom.jpg'
    test_img = './watch_data/Images_test' + '/' + 'kitchen.jpg'
    test_img = '/home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier/watch_dataset/Image/test/0/20230426121804.jpg'
    test_img = '/home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier/watch_dataset/Image/test/0/20230425115416.jpg'
    test_img = '/home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier/watch_dataset/Image/test/0/20230425115410.jpg'
    print(test_img)

    image = Image.open(test_img).convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "Livingroom(WatchTV).jpg").convert('RGB').resize((width, height))


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
    classification_label = labels[label_id]
    print("Image Label is :", classification_label, ", with Accuracy :", np.round(prob*100, 2), "%.")



def test2():
    data_folder = "./"
    data_folder_image = "./room_samples_0831/"
    data_folder_image = "./room_testset/2/"  # kitchen
    data_folder_image = "./room_testset/5/"  # door
    data_folder_image = "./room_testset/1/"  # bed
    data_folder_image = "./room_testset/3/"  # livingroom


    #model_path = "./home_model.tflite"
    model_path = "./home_default_model.tflite"
    model_path = "./ascc_efficientNet_model_default.tflite"

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

    # image = Image.open(data_folder_image + "Bedroom.jpg").convert('RGB').resize((width, height))
    #image = Image.open(data_folder_image + "Kitchen.jpg").convert('RGB').resize((width, height))
    # image = Image.open(data_folder_image + "Livingroom(WatchTV).jpg").convert('RGB').resize((width, height))


    import sys, random
    from pathlib import Path
    from PIL import Image
    import matplotlib.pyplot as plt


    # Retreive 9 random images from directory

    files=Path(data_folder_image).resolve().glob('*.*')
    test_sample= get_file_count_of_dir(data_folder_image)

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
            image = Image.open(img).convert('RGB').resize((width, height))
            # Classify the image.
            time1 = time.time()
            label_id, prob = classify_image(interpreter, image)
            print("label id:", label_id)
            time2 = time.time()
            classification_time = np.round(time2-time1, 3)
            print("Classificaiton Time =", classification_time, "seconds.")

                # Return the classification label of the image.
            classification_label = labels[label_id]
            print("file:", img)
            print("Image Label is :", classification_label, ", with Accuracy :", np.round(prob*100, 2), "%.")
            print("--------------------------------------------------------------------------------")

            plt.subplot(rows,cols,num+1)
            plt.title("Pred: "+classification_label + '(' + str(prob) + ')')
            # print("Pred: "+classification_label + '(' + str(prob) + ')')
            plt.axis('off')
            img = Image.open(img).convert('RGB')
            plt.imshow(img)
            plt.savefig("test_res.png")


            predict_res.append(classification_label)

    return predict_res



    # Read class labels.
    # labels = load_labels(label_path)




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
    dir = './dataset_online/test/'  # ./tf_testset
    dir = '/home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier/watch_dataset/Image/test/'
    path = dir
    y_test = []
    y_pred = []

    class_names=['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']

    accurate_count = 0
    total_count = 0
    
    for room in class_names:
        print(dir + '/' + room)
        fn = str(class_names.index(room))

        if os.path.isdir(dir + '/' + fn):
            print(fn)
            test_dir = dir + '/' + fn
            sample_cnt = get_file_count_of_dir(test_dir)
            print('sample cnt:', sample_cnt)
            total_count += sample_cnt
            
            test_truth_label = []
            for i in range(sample_cnt):
                index = int(fn)
                print('test lable:', class_names[index])
                test_truth_label.append(class_names[index])
                

            y_test.extend(test_truth_label)
            print('len test_truth_label:', len(test_truth_label))
            print('len y_test:', len(y_test))
            print('test_dir:', test_dir)

            predict_res = test_confusion_matrix(test_dir)
            print('len(predict_res):', len(predict_res))
            for index in range(len(predict_res)):
                if predict_res[index] == test_truth_label[index]:
                    accurate_count += 1
            y_pred.extend(predict_res)
        break


    import matplotlib.pyplot as plt
    # confusion matrix
    from mlxtend.plotting import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix

    print('y_pred len:', len(y_pred))
    print('y_pred:', y_pred)
    print('y_test len:', len(y_test))
    print('y_test:', y_test)

    print('accurate_count:', accurate_count, ' total_count:', total_count)
    print('test accuracy:', accurate_count * 1.0 / total_count)

    plt.figure()
    mat = confusion_matrix(y_test, y_pred, labels=class_names)
    cm = plot_confusion_matrix(conf_mat=mat, class_names=class_names, show_normed=True, figsize=(7,7))
    plt.show()
    #plt.savefig("room_cm_tflite_online_mnetv2.png")
    plt.savefig("room_cm_tflite_alexnet.png")


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
    #model_path = "./home_default_model.tflite"
    #model_path = "./home_16_size_model.tflite"
    #model_path = "./home_model_default_onlinedataset.tflite"
    #model_path = "./ascc_efficientNet_model_default.tflite"
    #model_path = "./ascc_mobilev2_model_default.tflite"

    global model_path

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
            print('file name:', file)
            label = predict(file, interpreter, class_names, width, height)
            predict_res.append(label)

    return predict_res


test()
#test2()

#get_confusion_matrix()
