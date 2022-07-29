# -*- coding: utf-8 -*-
"""
@author: Jason Zhang
@github: https://github.com/JasonZhang156/Sound-Recognition-Tutorial
"""

from keras.models import load_model
import tensorflow as tf
import esc10_input
import numpy as np
import os
import time

labels = ['door_open_closed', 'eating', 'keyboard', 'pouring_water_into_glass', 'toothbrushing', 'vacuum',
 'drinking', 'flush_toilet', 'microwave', 'quiet', 'tv_news', 'washing_hand']

def use_gpu():
    """Configuration for GPU"""
    from tensorflow.compat.v1.keras.backend import set_session
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    # config = tf.ConfigProto()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    set_session(tf.compat.v1.InteractiveSession(config=config))


def CNN_test(test_fold, feat):
    """
    Test model using test set
    :param test_fold: test fold of 5-fold cross validation
    :param feat: which feature to use
    """
    model = load_model('./cnn_logmel_foldone_second.h5')

    # 读取测试数据
    # _, _, test_features, test_labels = esc10_input.get_data(test_fold, feat)
    t1 = time.time()
    t2= time.time()
    while(1):
        try:
            t2= time.time()
            path = './mic_data/output.wav'
            # path = '/home/ascc/Downloads/total_sound_downsampling/cutting_food/cutting_food@4_freesound-cutting_food__538305__sound-2425__chopping-cutting.wav'
            test_feature = esc10_input.get_single_data(path)
            test_feature = np.expand_dims(test_feature, axis=-1)
            mean = np.mean(test_feature)
            std = np.std(test_feature)
            test_feature = (test_feature - mean) / std
            test_feature = test_feature.reshape(1,64,138,1)
            #print("test_feature: ",test_feature.shape)
            # 导入训练好的模型
            result = model.predict(test_feature)[0]
            # print(result)
            pre = labels[np.argmax(result)]
            val = result[np.argmax(result)]
            # print(pre,val)
            print(pre)
        except:
            print("ERROR........")

        # print("time (s) used: ",t2 - t1)
        # t1 = t2
        # print('\n')

    # 输出训练好的模型在测试集上的表现
    # score = model.evaluate(test_features, test_labels)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

    return result


if __name__ == '__main__':
    use_gpu()  # 使用GPU

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
