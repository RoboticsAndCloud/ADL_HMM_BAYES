"""Trains a DQN/DDQN to solve CartPole-v0 problem


"""

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import argparse
import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt


def motion_feature_extractor():
    motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)

    motion_id = tools_ascc.get_key(MOTION_ACTIVITY_MAPPING, motion_type)

    class_vector =[motion_id]
    print(class_vector)

    # Applying the function on input class vector
    from keras.utils import to_categorical
    output_matrix = to_categorical(class_vector, num_classes = 6, dtype ="int32")

    print(output_matrix)
    # [[0 0 0 0 0 1]]


    return output_matrix



MOTION_ACTIVITY_MAPPING = {
    0: 'jogging',
    1: 'jumping',
    2: 'laying',
    3: 'sitting',
    4: 'standing',
    5: 'walking'
}

motion_feature = [0, 0, 0, 0, 0, 1]
battery_feature = [1, 3]
adl_hidden_feature = [1, 2, 4, 5, 5, 5]

# features = motion_feature
# features.extend(battery_feature)
state = motion_feature + battery_feature + motion_feature
print("features:", state)

# from sklearn.preprocessing import LabelEncoder
# import numpy as np

# class_vector =['jogging','jumping', 'laying', 'sitting', 'standing', 'walking']

# code = np.array(class_vector)

# label_encoder = LabelEncoder()
# vec = label_encoder.fit_transform(code)
# print(vec)
# exit(0)

# array([ 2,  6,  7,  9, 19,  1, 16,  0, 17,  0,  3, 10,  5, 21, 11, 18, 19,
#         4, 22, 14, 13, 12,  0, 20,  8, 15])

# class_vector =['jogging','jumping', 'laying', 'sitting', 'standing', 'walking']
# print(class_vector)

# # Applying the function on input class vector
# from keras.utils import to_categorical
# output_matrix = to_categorical(class_vector)

# print(output_matrix)
motion_id = 4
class_vector =[5]
class_vector =[motion_id]
print(class_vector)

# Applying the function on input class vector
from keras.utils import to_categorical
output_matrix = to_categorical(class_vector, num_classes = 6, dtype ="int32")

print(output_matrix)
# [[0 0 0 0 0 1]]

exit(0)

def plot(rewards):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    # plt.legend()
    plt.savefig('reward.png')

    # plt.show()
    plt.clf()