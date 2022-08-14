# -*- coding: utf-8 -*-
"""
@author: Jason Zhang
@github: https://github.com/JasonZhang156/Sound-Recognition-Tutorial
"""

import numpy as np
from glob import glob
import os
import librosa
import feature_extraction as fe
import warnings
warnings.filterwarnings('ignore')

labels = ['door_open_closed', 'eating', 'keyboard', 'pouring_water_into_glass', 'toothbrushing', 'vacuum',
 'drinking', 'flush_toilet', 'microwave', 'quiet', 'tv_news', 'washing_hand']

# labels = ['door_open_closed', 'eating', 'keyboard', 'pouring_water_into_glass', 'vacuum',
#  'drinking', 'flush_toilet', 'quiet', 'tv_news', 'washing_hand']

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]

def extract_esc10_feat():
    # 5-fold cross validation settings
    # cv_index = np.load('cvindex.npz')

    # TR = cv_index['TR']  # train fold
    # TE = cv_index['TE']  # test fold
    # print(TR)
    # print(TE)


    '''
    flush_toilet@9_freesound-flush_toilet__253105__va7mjl__toilet-flush_2.wav
flush_toilet@9_freesound-flush_toilet__253105__va7mjl__toilet-flush_3.wav
flush_toilet@9_freesound-flush_toilet__253105__va7mjl__toilet-flush_4.wav
(base) ascc@ascc-XPS-8940:~/Downloads/total_sound_1_second/flush_toilet$ pwd

/home/ascc/Downloads/total_sound_1_second/flush_toilet

    '''
    info = {}
    for i in labels:
        sub_dir = "/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/ascc_activity_1second/audio/" + i
        files = os.listdir(sub_dir)
        aa = []
        for a_file in files:
            aa.append(sub_dir+"/"+a_file)
        info[i] = aa
    # print(info)
    # 5 fold extracting
    a = 0
    # for fold in range(1, 6):
    # print('Extracting training set and test set for fold {}'.format(fold))

    train_feats = []
    train_labels = []
    test_feats = []
    test_labels = []

    # class_list = np.sort(glob('./data/esc10/audio/*'))
    # class_list = np.sort(glob('/home/ascc/Downloads/total_sound_1_second/*'))

    # for a_label, in labels:
    for index, a_label in enumerate(labels):
        paths = info[a_label]
        num_files = len(paths)
        down = 0
        up = int(num_files*0.8)
        print('up:', up)
        for i in range(down, up):
            #try:
            train_audio = paths[i]
            
            y, fs = librosa.load(train_audio, sr=44100,mono=True)
            y = pad_truncate_sequence(y, 44100)
            # feat = fe.extract_mfcc(y, fs, 3)
            feat = fe.extract_logmel(y, fs, 1)
            # print('len feat:', len(feat[0]))
            train_feats.append(feat)
            train_labels.append(index)
            a =a +1
            #except:
                #print("error in train_audio", train_audio)
        print('len train_feat:', len(train_feats))
        # for test
        down = int(num_files*0.8)
        up = num_files
        # test_audio = audio_list[TE[fold - 1]]
        for j in range(down, up):
            #try:
            test_audio = paths[j]
            
            # print('Processing sound class: ', os.path.basename(classpath), index+1, '/', len(class_list),
            #       ' --- test set: ', j+1, '/', len(test_audio))
            y, fs = librosa.load(test_audio, sr=44100, mono=True)
            y = pad_truncate_sequence(y, 44100)
            # feat = fe.extract_mfcc(y, fs, 3)
            feat = fe.extract_logmel(y, fs)
            test_feats.append(feat)
            test_labels.append(index)
            a =a +1
            #except:
                #print("error in test_audio", test_audio)
    print("total sample: ",a)
    train_feats = np.array(train_feats)
    train_labels = np.array(train_labels)
    test_feats = np.array(test_feats)
    test_labels = np.array(test_labels)

    # todo: check the shape, how to get the shape

    print('train_feats:', train_feats.shape)
    print('test_feat:', test_feats.shape)
    print('test_labes:', test_labels.shape)
    print('test_labes:',test_labels)

    # np.savez('./data/esc10/feature/esc10_mfcc_fold{}.npz'.format(fold),
    #          train_x=train_feats, train_y=train_labels, test_x=test_feats, test_y=test_labels)
    np.savez('/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/ascc_activity_1second/feature/ascc_logmel_total.npz',
                train_x=train_feats, train_y=train_labels, test_x=test_feats, test_y=test_labels)


if __name__ == '__main__':
    extract_esc10_feat()
