"""Trains a DQN/DDQN to solve CartPole-v0 problem


"""
#import ground_truth_dict_dataset0819


import json

def save_pet(pet, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(pet))

def load_pet(filename):
    with open(filename) as f:
        pet = json.loads(f.read())
    return pet


import matplotlib.pyplot as plt


def plotone(rewards, fig = "test.png", xlabel = "Episodes", ylabel="Rewards"):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend()
    plt.savefig(fig)

    #plt.show()
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

# def motion_feature_extractor():
#     motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)

#     motion_id = tools_ascc.get_key(MOTION_ACTIVITY_MAPPING, motion_type)

#     class_vector =[motion_id]
#     print(class_vector)

#     # Applying the function on input class vector
#     from keras.utils import to_categorical
#     output_matrix = to_categorical(class_vector, num_classes = 6, dtype ="int32")

#     print(output_matrix)
#     # [[0 0 0 0 0 1]]


#     return output_matrix



# MOTION_ACTIVITY_MAPPING = {
#     0: 'jogging',
#     1: 'jumping',
#     2: 'laying',
#     3: 'sitting',
#     4: 'standing',
#     5: 'walking'
# }

# # r1 = [1,2,3,4,5,6]
# # r2 = [3,4,5,6,7]
# # plot(r1, r2)

# motion_feature = [0, 0, 0, 0, 0, 1]
# battery_feature = [1, 3]
# adl_hidden_feature = [1, 2, 4, 5, 5, 5]

# features = motion_feature
# features.extend(battery_feature)
# state = motion_feature + battery_feature + motion_feature
# print("features:", state)

# # from sklearn.preprocessing import LabelEncoder
# # import numpy as np

# # class_vector =['jogging','jumping', 'laying', 'sitting', 'standing', 'walking']

# # code = np.array(class_vector)

# # label_encoder = LabelEncoder()
# # vec = label_encoder.fit_transform(code)
# # print(vec)
# # exit(0)

# # array([ 2,  6,  7,  9, 19,  1, 16,  0, 17,  0,  3, 10,  5, 21, 11, 18, 19,
# #         4, 22, 14, 13, 12,  0, 20,  8, 15])

# # class_vector =['jogging','jumping', 'laying', 'sitting', 'standing', 'walking']
# # print(class_vector)

# # # Applying the function on input class vector
# # from keras.utils import to_categorical
# # output_matrix = to_categorical(class_vector)

# # print(output_matrix)
# motion_id = 4
# class_vector =[5]
# class_vector =[motion_id]
# print(class_vector)

# # Applying the function on input class vector
# from keras.utils import to_categorical
# output_matrix = to_categorical(class_vector, num_classes = 6, dtype ="int32")

# print(output_matrix)
# # [[0 0 0 0 0 1]]

# class_vector = [[1, 2, 3, 4, 5],[1, 3, 8, 10, 14]
#                 ]
# output_matrix = to_categorical(class_vector)
# # print(output_matrix)

# class_vector = [[[1], [2], [3], [4], [5]],[[1], [3], [8], [10], [14]]
#                 ]
# output_matrix = to_categorical(class_vector)
# print(output_matrix)

# # exit(0)

# motion_feature = list(output_matrix[0])
# state = motion_feature + battery_feature + motion_feature

# print("features:", state)
# print(type(state))

# state = np.reshape(state, [1, len(state)])
# print("features:", state)
# print("features size:", state.size)

# WMU_audio = "WMU_audio"
# WMU_vision = "WMU_vision"
# WMU_fusion = WMU_audio + WMU_vision
# Robot_audio_vision = "Robot_fusion"
# Robot_WMU_audio = Robot_audio_vision + WMU_audio
# Robot_WMU_vision = Robot_audio_vision + WMU_vision
# Robot_WMU_fusion = Robot_audio_vision + WMU_fusion

# Nothing = "Nothing"

# RL_ACTION_DICT = {
#     0: WMU_audio,  
#     1: WMU_vision, 
#     2: WMU_fusion,  
#     3: Robot_audio_vision,
#     4: Robot_WMU_audio, # robot and WMU both capture data
#     5: Robot_WMU_vision,
#     6: Robot_WMU_fusion,
#     7: Nothing
# }

# action_space = list(RL_ACTION_DICT.keys())
# print(action_space)
# print(random.sample(action_space,1))


# thisdict =	{
#   "brand": "Ford",
#   "model": "Mustang",
#   "year": 1964
# }
# thisdict["color"] = "red"
# print(thisdict)

# di = {}
# di[1] = 2
# di[2] = (3,4)
# print(di)

# a, b = di[2]
# print(a)
# print(b)

# k = 2
# if k in di.keys():
#     print(di[k])


# image_dir_name = "/home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES_V2/ADL_HMM_BAYES/Ascc_Dataset_0819//Image/2009-12-11-08-46-27/"
# target_time_str = image_dir_name.split('Image/')[1].rstrip('/')
# print(target_time_str)

# timestamp = 10

# for i in range(timestamp, 20-5):
#     print("i:",)


# res_dict = {}
# res_dict['kitchen'] = 0.2
# res_dict['bedroom'] = 0.3
# res_dict['bathroom'] = 0.1
# res_dict['livingroom'] = 0.4

# print("len:", len(res_dict))

# def sorter_take_count(elem):
#     # print('elem:', elem)
#     return elem[1]

# sd = sorted(res_dict.items(), key=sorter_take_count, reverse=True)


# for k,v in sd:
#     print('res2:', k, ' v:', v)
#     # break


# scores = [-1048.499999999825, -854.9999999998544, -129.20000000006905, 509.6799999999743, 175.85999999994436, -243.05999999998932, -655.4000000000583, -71.8600000000444, 1548.6599999999617, 1322.9000000000174]

# plotone(scores)







def sorter_dict(elem):
    # print('elem:', elem)
    return elem[0]

#ground_truth_dict = ground_truth_dict_dataset0819.ground_truth_dict
#
#sd = sorted(ground_truth_dict.items(), key=sorter_dict, reverse=True)
#
#for k,v in sd:
#    print(k, ' ', v)
#
#class_vector =[1000]
#print(class_vector)
#
## Applying the function on input class vector
#from keras.utils import to_categorical
#output_matrix = to_categorical(class_vector, num_classes = 1001, dtype ="int32")
#
#print(output_matrix)
# [[0 0 0 0 0 1]]







scores = [-1925.4199999996893, -1189.3199999999026, -676.9599999999662, -417.17999999995845, -162.57999999999632, -0.20000000000198725, -8.250000000000147, 40.15000000000398, 224.36000000000314, 330.40000000000856, 104.03999999999876, 298.76000000000033, 2.2000000000001556, 88.09999999999576, 45.50000000000874, 338.2400000000048, 364.06000000000626, 385.6600000000058, 341.10000000000883, 382.1799999999979, 341.9600000000184, 395.75999999999885, 342.8000000000016, 402.7799999999981, 356.6699999999993, 386.73000000000553, 386.7799999999947, 417.5200000000011, 414.1700000000009, 400.070000000002, 365.0299999999984, 430.86999999999733, 410.5200000000035]

print('len(scores):', len(scores))
plotone(scores, fig = "dqn_accumulated_rewards.png", xlabel = "Episodes", ylabel="Accumulated Rewards")

total_wmu_cam_trigger_times = [4782, 3413, 2263, 1700, 854, 817, 398, 354, 467, 302, 142, 180, 290, 37, 58, 141, 106, 86, 115, 72, 107, 82, 90, 78, 69, 82, 83, 71, 85, 75, 75, 72, 91]
total_robot_trigger_times = [6002, 5983, 6002, 6001, 6002, 6002, 6002, 6002, 5422, 2727, 6002, 5887, 6002, 6002, 6002, 5248, 4388, 4766, 4779, 3544, 2224, 2808, 4733, 2938, 3670, 3534, 2754, 1870, 1818, 1212, 2241, 933, 686]
total_privacy_times = [783, 566, 335, 319, 195, 168, 113, 105, 127, 100, 56, 85, 27, 38, 15, 88, 72, 70, 67, 74, 81, 69, 72, 69, 80, 73, 64, 80, 67, 65, 80, 68, 63]


print('len(wmu times):', len(total_wmu_cam_trigger_times))

plt.figure(figsize=(20,5))
plt.plot(total_wmu_cam_trigger_times, label='wmu_cam_trigger_times')
plt.plot(total_robot_trigger_times, label='total_robot_trigger_times')
plt.plot(total_privacy_times, label='privacy_occur_times')

plt.xlabel("Episodes")
plt.ylabel("Times")
plt.legend()
plt.savefig('dqn_multi_reward.png')

# plt.show()
plt.clf()

exit(0)

