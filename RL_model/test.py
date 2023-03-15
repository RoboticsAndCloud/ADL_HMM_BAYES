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


def plot(rewards):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    # plt.legend()
    plt.savefig('reward.png')

    # plt.show()
    plt.clf()

def plot(rewards1, rewards2):
    plt.figure(figsize=(20,5))
    plt.plot(rewards1, label='rewards1')
    plt.plot(rewards2, label='rewards2')

    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig('multi_reward.png')

    plt.show()
    plt.clf()

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

# r1 = [1,2,3,4,5,6]
# r2 = [3,4,5,6,7]
# plot(r1, r2)

motion_feature = [0, 0, 0, 0, 0, 1]
battery_feature = [1, 3]
adl_hidden_feature = [1, 2, 4, 5, 5, 5]

features = motion_feature
features.extend(battery_feature)
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

class_vector = [[1, 2, 3, 4, 5],[1, 3, 8, 10, 14]
                ]
output_matrix = to_categorical(class_vector)
# print(output_matrix)

class_vector = [[[1], [2], [3], [4], [5]],[[1], [3], [8], [10], [14]]
                ]
output_matrix = to_categorical(class_vector)
print(output_matrix)

# exit(0)

motion_feature = list(output_matrix[0])
state = motion_feature + battery_feature + motion_feature

print("features:", state)
print(type(state))

state = np.reshape(state, [1, len(state)])
print("features:", state)
print("features size:", state.size)

WMU_audio = "WMU_audio"
WMU_vision = "WMU_vision"
WMU_fusion = WMU_audio + WMU_vision
Robot_audio_vision = "Robot_fusion"
Robot_WMU_audio = Robot_audio_vision + WMU_audio
Robot_WMU_vision = Robot_audio_vision + WMU_vision
Robot_WMU_fusion = Robot_audio_vision + WMU_fusion

Nothing = "Nothing"

RL_ACTION_DICT = {
    0: WMU_audio,  
    1: WMU_vision, 
    2: WMU_fusion,  
    3: Robot_audio_vision,
    4: Robot_WMU_audio, # robot and WMU both capture data
    5: Robot_WMU_vision,
    6: Robot_WMU_fusion,
    7: Nothing
}

action_space = list(RL_ACTION_DICT.keys())
print(action_space)
print(random.sample(action_space,1))


thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict["color"] = "red"
print(thisdict)

di = {}
di[1] = 2
di[2] = (3,4)
print(di)

a, b = di[2]
print(a)
print(b)

k = 2
if k in di.keys():
    print(di[k])


image_dir_name = "/home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES_V2/ADL_HMM_BAYES/Ascc_Dataset_0819//Image/2009-12-11-08-46-27/"
target_time_str = image_dir_name.split('Image/')[1].rstrip('/')
print(target_time_str)

timestamp = 10

for i in range(timestamp, 20-5):
    print("i:",)


res_dict = {}
res_dict['kitchen'] = 0.2
res_dict['bedroom'] = 0.3
res_dict['bathroom'] = 0.1
res_dict['livingroom'] = 0.4

print("len:", len(res_dict))

def sorter_take_count(elem):
    # print('elem:', elem)
    return elem[1]

sd = sorted(res_dict.items(), key=sorter_take_count, reverse=True)


for k,v in sd:
    print('res2:', k, ' v:', v)
    # break

exit(0)

