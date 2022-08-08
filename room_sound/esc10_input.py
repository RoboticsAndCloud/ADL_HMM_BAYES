# -*- coding: utf-8 -*-
"""
@author: Jason Zhang
@github: https://github.com/JasonZhang156/Sound-Recognition-Tutorial
"""

import numpy as np
import librosa
import feature_extraction as fe

RANDOM_SEED = 20181212
def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=int)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def get_single_data(file_path):
    # y, fs = librosa.load(file_path, sr=32000,mono=True)
    # y = pad_truncate_sequence(y, 32000*5)
    # feat = fe.extract_logmel(y, fs, 5)


    y, fs = librosa.load(file_path, sr=44100,mono=True)
    y = pad_truncate_sequence(y, 44100)
    train_feats = fe.extract_logmel(y, fs, 1)
    # print("train_feats: ", train_feats.shape)
    
    return train_feats



def get_data(test_fold, feat):
    """load feature for train and test"""
    # load feature
    data = np.load('./data/esc10/feature/esc10_{}_fold{}.npz'.format(feat, test_fold))
    train_x = np.expand_dims(data['train_x'], axis=-1)
    train_y = data['train_y']
    test_x = np.expand_dims(data['test_x'], axis=-1)
    test_y = data['test_y']

    # one-hot encode
    train_y = dense_to_one_hot(train_y, 33)
    test_y = dense_to_one_hot(test_y, 33)

    # z-score normalization
    mean = np.mean(train_x)
    std = np.std(train_x)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    # shuffle
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(train_x)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(train_y)

    print('Audio Feature: ', feat)
    print('Training Set Shape: ', train_x.shape)
    print('Test Set Shape: ', test_x.shape)

    return train_x, train_y, test_x, test_y


def get_data_all(test_fold, feat, number_class=12):
    """load feature for train and test"""
    # load feature


    # for i in [1,2,3,4,5]:
    file_dir = test_fold
    data = np.load(file_dir)
    print(data['test_y'])
    
    train_x=np.expand_dims(data['train_x'], axis=-1)
    train_y=data['train_y']
    test_x=np.expand_dims(data['test_x'], axis=-1)
    test_y=data['test_y']

    # one-hot encode
    print("train_y: ",len(train_y))
    print("train_x: ",len(test_y))

    train_y = dense_to_one_hot(train_y, 12)
    test_y = dense_to_one_hot(test_y, 12)

    # z-score normalization
    mean = np.mean(train_x)
    std = np.std(train_x)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    # shuffle
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(train_x)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(train_y)

    print('Audio Feature: ', feat)
    print('Training Set Shape: ', train_x.shape)
    print('Test Set Shape: ', test_x.shape)

    return train_x, train_y, test_x, test_y