# -*- coding: utf-8 -*-
"""
@author: Jason Zhang
@github: https://github.com/JasonZhang156/Sound-Recognition-Tutorial
"""

#from keras.models import load_model
#import tensorflow as tf
import esc10_input
import numpy as np
import os
import time
import logging
from timeit import default_timer as timer

labels = ['door_open_closed', 'eating', 'keyboard', 'pouring_water_into_glass', 'toothbrushing', 'vacuum', 'drinking', 'flush_toilet', 'microwave', 'quiet', 'tv_news', 'washing_hand']
# labels = ['door_open_closed', 'eating', 'keyboard', 'pouring_water_into_glass', 'vacuum', 'drinking', 'flush_toilet', 'quiet', 'tv_news', 'washing_hand'] # 10

ASCC_DATA_NOTICE_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/ascc_data/notice.txt'
ASCC_DATA_RES_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/ascc_data/recognition_result.txt'

ob_folder = '/home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/room_sound/sound_dataset/ascc_activity_1second/feature/ascc_logmel_total.npz'
num_class = len(labels)
mean, std = esc10_input.get_mean_std(ob_folder, 'logmel',num_class)


def read_dir_name(file_name):
    with open(file_name, 'r') as f:
        dir_name = str(f.read().strip())
        f.close()
    print('dir_name:%s', dir_name)
    return dir_name

def write_res_into_file(file_name, res_list):
    with open(file_name, 'w') as f:
        f.write(str(res_list))
        f.write('\t')
        f.close()
    
    return True

def use_gpu():
    """Configuration for GPU"""
    from tensorflow.compat.v1.keras.backend import set_session
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    # config = tf.ConfigProto()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    set_session(tf.compat.v1.InteractiveSession(config=config))

MOTION_TFLITE_MODEL = './sound_default_model.tflite'
from tflite_runtime.interpreter import Interpreter

def predict_tflite(test_data):


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
    # test_sound = np.expand_dims(test_data, axis=0).astype(np.float32)
    test_sound = np.float32(test_data)
    print("test_sound shape:", test_sound.shape)


    interpreter.set_tensor(input_index, test_sound)

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
    print('prediction_classes:', prediction_classes, " ", labels[prediction_classes])
    pred = labels[prediction_classes]

    return pred, output[prediction_classes]



def CNN_test(test_fold, feat):
    """
    Test model using test set
    :param test_fold: test fold of 5-fold cross validation
    :param feat: which feature to use
    """
    #model = load_model('./cnn_logmel_foldone_second.h5')

    # 读取测试数据
    # _, _, test_features, test_labels = esc10_input.get_data(test_fold, feat)
    t1 = time.time()
    t2= time.time()

    pre_test_dir = ''
    while(True):
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

        start = timer()

        try:
            t1= time.time()
            path = test_dir + '/' + 'recorded.wav'
            # path = './mic_data/output.wav'
            # path = '/home/ascc/Downloads/total_sound_downsampling/cutting_food/cutting_food@4_freesound-cutting_food__538305__sound-2425__chopping-cutting.wav'
            test_feature = esc10_input.get_single_data(path)
            test_feature = np.expand_dims(test_feature, axis=-1)

            test_feature = (test_feature - mean) / std
            test_feature = test_feature.reshape(1,64,138,1)
            #print("test_feature: ",test_feature.shape)
            # 导入训练好的模型
            pre, val = predict_tflite(test_data=test_feature)

            print('pre:', pre, ' val:', val)
            print(pre + '(' + str(val) + ')')
            t2 = time.time()
            print("Get_prediction time cost try:", np.round(t2-t1, 3))   

            
            res = pre + '(' + str(val) + ')'
        except Exception as e:
            print("ERROR........", e)
            res = ''

    
        end = timer()
        print("Get_prediction time cost:", end-start)   

        write_res_into_file(ASCC_DATA_RES_FILE, res)            


        # print("time (s) used: ",t2 - t1)
        # t1 = t2
        # print('\n')

    # 输出训练好的模型在测试集上的表现
    # score = model.evaluate(test_features, test_labels)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

    return result


if __name__ == '__main__':
    #use_gpu()  # 使用GPU

    acc = CNN_test(1, 'logmel')

    # dict_acc = {}
    # print('### [Start] Test model for ESC10 dataset #####')
    # for fold in [1, 2, 3, 4, 5]:
    #     print("## Start test fold{} model #####".format(fold))
    #     acc = CNN_test(fold, 'mfcc')
    #     dict_acc['fold{}'.format(fold)] = acc
    #     print("## Finish test fold{} model #####".format(fold))
    # dict_acc['mean'] = np.mean(list(dict_acc.values()))
    # print(dict_acc)
    # print('### [Finish] Test model finished for ESC10 dataset #####')
