"""
Brief: Bayes model for activity probability, including vision, motion, audio-based model.
Author: Frank
Date: 07/10/2022
"""

from datetime import datetime
from pickle import TRUE
import random
from re import T
from collections import deque
import gc
# Applying the function on input class vector
from keras.utils import to_categorical



import collections

#import constants
#from tkinter.messagebox import NO

import hmm
import rl_env_ascc
import motion_adl_bayes_model
import tools_ascc
import constants

# import tools_sql
import matplotlib.pyplot as plt
import numpy as np

import time_adl_res_dict

from timeit import default_timer as timer

#from tensorflow.keras.utils import to_categorical

from mem_top import mem_top
print(mem_top(limit=15,width=180))


OUTPUT_RANK = False

MILAN_BASE_DATE = '2009-10-16'

MILAN_BASE_DATE_HOUR = '2009-10-16 06:00:00'

TEST_BASE_DATE = '2009-12-11'
DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
HOUR_TIME_FORMAT = "%H:%M:%S"
DAY_FORMAT_STR = '%Y-%m-%d'

UNCERTAIN_CHECK_INTERVAL = 60 * 2 # Seconds

LIVING_ROOM_CHECK_TIMES_MAX = 2

new_activity_check_times = 2
DOUBLE_CHECK = 2

AUDIO_WEIGHT = 0.6

LOCATION_DIR_SPLIT_SYMBOL = ':'

BATHROOM = 'athroom'

new_activity_factor = 1.0 # when detect new activity, reward more 
MOTION_TRANSITION_REWARD = 1 # when detect the walking transition, reward + 1

"""
Given the duration, return the probability that the activity may be finished
For Example: Read, we got 20mins(5 times), 30mins(10), 40mins(25), 60mins(2),  for duration of 20mins, the probability would be 5/(5+10+25+2) = 11.90% 
"""
# act_duration_cnt_dict = tools_ascc.get_activity_duration_cnt_set()
# def get_end_of_activity_prob_by_duration(activity_duration, activity):
#     d_lis = act_duration_cnt_dict[activity]
#     total_cnt = len(d_lis)
#     cnt = 0
#     for d in d_lis:
#         if activity_duration >= d:
#             cnt += 1
    
#     prob = 1 - cnt * 1.0 /total_cnt

#     if prob < 0.01:
#         prob = 0.01

#     return prob

def plotone(rewards, figure = "reward.png"):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    # plt.legend()
    plt.savefig(figure)

    # plt.show()
    plt.clf()

def plot(rewards1, rewards2, label1 = "r1", label2 = "r2"):
    plt.figure(figsize=(20,5))
    plt.plot(rewards1, label=label1)
    plt.plot(rewards2, label=label2)

    plt.xlabel("Episode")
    plt.ylabel("Times")
    plt.legend()
    plt.savefig('multi_trigger_times.png')

    # plt.show()
    plt.clf()


def sorter_take_count(elem):
    # print('elem:', elem)
    return elem[1]

def sorter_dict(elem):
    # print('elem:', elem)
    return elem[0]

def get_hmm_model():
    state_list, symbol_list = tools_ascc.get_activity_for_state_list()
    sequences = []
    
    for i in range(len(state_list) -15):
        print(state_list[i])
        print("==")
        seq = (state_list[i], symbol_list[i])
        sequences.append(seq)
        
    print('len sequence:', len(sequences))
    print(sequences[1])

    model = hmm.train(sequences, delta=0.001, smoothing=0)

    print('model._states:', model._states)
    print('model._symbols:', model._symbols )
    print('model._start_prob:', model._start_prob)
    print('model._trans_prob:', model._trans_prob)
    print('model._emit_prob:', model._emit_prob)

    return model


day_time_str = '2009-12-11'
day_begin ='08:45:00'


activity_date_dict, activity_begin_dict, activity_end_dict, \
        activity_begin_list, activity_end_list  = tools_ascc.get_activity_date(day_time_str)

# from milan dataset
def get_activity_by_time_str2(activity_time_str):

    

    day_begin, day_end = tools_ascc.get_day_begin_end(activity_date_dict,
                                            activity_begin_dict, activity_end_dict)

    hit_activity_check_times = 0

    # print("=====================================")
    # print("Date:", day_time_str)
    # print("activity_begin_dict", len(activity_begin_dict))
    # print("activity_end_dict", len(activity_end_dict))

    motion_activity_cnt = 0
    output_dict = {}
    output_dict2 = {}  # timestamp is at the beginning

    output_miss_event_dict = {}

    last_hit_time_list = []
    last_miss_time_set = set()
    print("Activity", "\t", "Start", "\t", "End", "\t", "Duration")

    res = []
    res_dict = {}

    for key in activity_begin_dict.keys():
        time_list_begin = activity_begin_dict[key]
        time_list_end = activity_end_dict[key]
        if key == 'Sleep':
            time_list_begin = time_list_begin[:-1]
            time_list_end = time_list_end[1:]
        for t_i in range(len(time_list_begin)):
            a_begin = datetime.strptime(time_list_begin[t_i], tools_ascc.DATE_HOUR_TIME_FORMAT)
            try:
                a_end = datetime.strptime(time_list_end[t_i], tools_ascc.DATE_HOUR_TIME_FORMAT)
            except:
                print("End list not good", len(time_list_begin), len(time_list_end))
                break

            duration = (a_end - a_begin).seconds * 1.0 /60


            # each day start after getting up (sleep end), ignore the activies before the time of 'sleep end'
            tmp_a_end = datetime.strptime(time_list_begin[t_i].split()[1], HOUR_TIME_FORMAT)
            if tmp_a_end < day_begin:
                # print("A end < day begin, ignore:", a_end, day_begin)
                break

            motion_activity_cnt = motion_activity_cnt + 1

            if duration > 0:
                tmp_str = key + "\t" + time_list_begin[t_i] + "\t" + time_list_end[t_i] + "\t" + str(duration)
                tmp_str2 = time_list_begin[t_i] + "\t" + time_list_end[t_i] + "\t" + key + "\t" + str(duration)

                output_dict[time_list_begin[t_i]] = tmp_str
                output_dict2[time_list_begin[t_i]] = tmp_str2



                k_time = activity_time_str
                hit_time = datetime.strptime(k_time, tools_ascc.DATE_HOUR_TIME_FORMAT)
                if hit_time >= a_begin and hit_time <= a_end:
                    #print("key a_begin, a_begin, hit_time:",key, a_begin, a_end, hit_time)

                    hit_activity_check_times = hit_activity_check_times + 1
                    ## Note: in dict, the time is out off order
                    last_hit_time_list.append(hit_time)
                    # print("#####:hit time:", hit_time, key)
                    # return key
                    res.append((key, a_begin, a_end, hit_time))
                    res_dict[key] = a_begin

    print(res)

    if len(res) == 1:
        return res[0][0]
    elif len(res) > 1:
        sd = sorted(res_dict.items(), key=sorter_take_count, reverse=True)
        for k,v in sd:
            return k

                
    print("Got empty hit_time:", hit_time)
    return ''

# test_time_str = '2009-12-11 09:10:30'
# print(get_activity_by_time_str(test_time_str))
# exit(0)


def get_object_by_activity(activity):
    # book, medicine, laptop, plates & fork & food, toilet
    print('object activity:', activity)
    act_dict = motion_adl_bayes_model.P4_Object_Under_Act[activity]
    print(act_dict)

    sd = sorted(act_dict.items(), key=sorter_take_count, reverse=True)
    res = sd[0][0]

    random_t = random.random()
    print('get_object_by_activity random_t:', random_t)

    if random_t > sd[0][1] and (len(sd) > 1):
        index = random.randint(1, len(sd)-1)
        res = sd[index][0]

    return res


# get object by CNN model
def get_object_by_activity_yolo(time_str):
    """
    # Location
    LOCATION_READINGROOM = 'readingroom'
    LOCATION_BATHROOM = 'bathroom'
    LOCATION_BEDROOM = 'bedroom'
    LOCATION_LIVINGROOM = 'livingroom'
    LOCATION_KITCHEN = 'Kitchen'
    LOCATION_DININGROOM = 'diningroom'
    LOCATION_DOOR = 'door'
    LOCATION_LOBBY = 'lobby'
    """
    # Mapping

    # should be act : probability
    # /home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/src/
    # ascc_room_activity_test.py
    object_dict = tools_ascc.get_activity_by_vision_yolov3(time_str, action='vision', mode='None-map')
    print('get_object_by_activity_yolo time_str:', time_str, ' object:', object_dict)

    return object_dict

def get_location_by_activity(activity):
    """
    # Location
    LOCATION_READINGROOM = 'readingroom'
    LOCATION_BATHROOM = 'bathroom'
    LOCATION_BEDROOM = 'bedroom'
    LOCATION_LIVINGROOM = 'livingroom'
    LOCATION_KITCHEN = 'Kitchen'
    LOCATION_DININGROOM = 'diningroom'
    LOCATION_DOOR = 'door'
    LOCATION_LOBBY = 'lobby'
    """
    # Mapping
    print('activity:', activity)
    act_dict = motion_adl_bayes_model.P1_Location_Under_Act[activity]

    sd = sorted(act_dict.items(), key=sorter_take_count, reverse=True)
    res = sd[0][0]

    random_t = random.random()
    print('get_location_by_activity random_t:', random_t)
    if random_t > sd[0][1] and (len(sd) > 1):
        index = random.randint(1, len(sd)-1)
        res = sd[index][0]

    return res


# get location by CNN model
def get_location_by_activity_cnn(time_str):
    """
    # Location
    LOCATION_READINGROOM = 'readingroom'
    LOCATION_BATHROOM = 'bathroom'
    LOCATION_BEDROOM = 'bedroom'
    LOCATION_LIVINGROOM = 'livingroom'
    LOCATION_KITCHEN = 'Kitchen'
    LOCATION_DININGROOM = 'diningroom'
    LOCATION_DOOR = 'door'
    LOCATION_LOBBY = 'lobby'
    """
    # Mapping

    # should be act : probability
    # /home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/src/
    # ascc_room_activity_test.py
    location, prob = tools_ascc.get_activity_by_vision_dnn(time_str, action='vision')
    print('get_location_by_activity_CNN time_str:', time_str, ' location:', location, ' prob:', prob)

    return location, float(prob)

def get_motion_type_by_activity(activity):
    # motion type: sitting, standing, walking, random by the probs

        # Mapping
    act_dict = motion_adl_bayes_model.P2_Motion_type_Under_Act[activity]

    sd = sorted(act_dict.items(), key=sorter_take_count, reverse=True)
    res = sd[0][0]

    random_t = random.random()
    print('get_motion_type_by_activity random_t:', random_t)
    if random_t > sd[0][1] and (len(sd) > 1):
        index = random.randint(1, len(sd)-1)
        res = sd[index][0]

    return res

def get_motion_type_by_activity_cnn(time_str):

    # TODO: get motion type for time_motion_type dict
    # Mapping
    # should be act : probability
    # /home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/src/
    # ascc_room_activity_test.py

    target_folder_time_str = get_target_folder_time_str(time_str)
    # print("target_folder_time_str:", target_folder_time_str, " time_str:", cur_time_str, ' location:', location)

    if target_folder_time_str in time_motion_dict.keys():
        motion_type, prob = time_motion_dict[target_folder_time_str]
        return motion_type, float(prob)

    # as we recognize all the motion action and store in the dict, here we just returl null
    return '', -1

    motion_type_list, prob_list = tools_ascc.get_activity_by_motion_dnn(time_str, action='vision')
    motion_type = ''
    prob = -1
    if len(motion_type_list) > 0:
        motion_type = motion_type_list[-1]
        prob = float(prob_list[-1])



    #time_motion_dict[target_folder_time_str] = (motion_type, prob)

    print('get_motion_type_by_activity_cnn time_str:', time_str, ' motion_type:', motion_type, ' prob:', prob)

    return motion_type, float(prob)

def get_audio_type_by_activity(activity):
    # audio type:
    # door_open_closed
    # drinking
    # eating
    # flush_toilet
    # keyboard
    # microwave
    # pouring_water_into_glass
    # quiet
    # toothbrushing
    # tv_news
    # vacuum
    # washing_hand

    # Mapping
    act_dict = motion_adl_bayes_model.P3_Audio_type_Under_Act[activity]

    sd = sorted(act_dict.items(), key=sorter_take_count, reverse=True)
    res = sd[0][0]

    random_t = random.random()
    print('get_audio_type_by_activity random_t:', random_t)
    if random_t > sd[0][1] and (len(sd) > 1):
        index = random.randint(1, len(sd)-1)
        res = sd[index][0]
    
    return res

def get_audio_type_by_activity_cnn(time_str):
    # audio type:
    # door_open_closed
    # drinking
    # eating
    # flush_toilet
    # keyboard
    # microwave
    # pouring_water_into_glass
    # quiet
    # toothbrushing
    # tv_news
    # vacuum
    # washing_hand

    # Mapping
    # should be act : probability
    # /home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/src/
    # ascc_room_activity_test.py
    audio_type, prob = tools_ascc.get_activity_by_audio_dnn(time_str, action='vision')

    print('get_audio_type_by_activity_cnn time_str:', time_str, ' audio_type:', audio_type, ' prob:', prob)

    return audio_type, float(prob)
    

MOTION_ACTIVITY_MAPPING = {
    0: 'jogging',
    1: 'jumping',
    2: 'laying',
    3: 'sitting',
    4: 'standing',
    5: 'walking'
}

def motion_feature_extractor(motion_type):
    # motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)

    motion_id = tools_ascc.get_key(MOTION_ACTIVITY_MAPPING, motion_type)

    class_vector =[motion_id]
    print(class_vector)

    output_matrix = to_categorical(class_vector, num_classes = 6, dtype ="int32")

    # print(output_matrix)
    # [[0 0 0 0 0 1]]

    return output_matrix[0]


def adl_hidden_feature_extractor(act):
    # motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)

    act_id = tools_ascc.get_key(tools_ascc.ACTIVITY_DICT, act)

    class_vector =[act_id]
    # print(class_vector)

    # Applying the function on input class vector
    #from keras.utils import to_categorical
    output_matrix = to_categorical(class_vector, num_classes = len(tools_ascc.ACTIVITY_DICT), dtype ="int32")

    return output_matrix[0]


env = rl_env_ascc.EnvASCC(TEST_BASE_DATE + ' 00:00:00')
env.reset()

hmm_model = get_hmm_model()

bayes_model_location = motion_adl_bayes_model.Bayes_Model_Vision_Location(hmm_model=hmm_model, simulation=False)
bayes_model_motion = motion_adl_bayes_model.Bayes_Model_Motion(hmm_model=hmm_model, simulation=True)
bayes_model_audio = motion_adl_bayes_model.Bayes_Model_Audio(hmm_model=hmm_model, simulation=False)
bayes_model_object = motion_adl_bayes_model.Bayes_Model_Vision_Object(hmm_model=hmm_model, simulation=True)


cur_activity_prob = 0
pre_activity = ''
cur_activity = ''
activity_begin_time = '2009-10-16 06:00:00'
activity_duration = 0

# TODO how to record transition activities
res_prob = {}
rank1_res_prob = []
rank2_res_prob = []
rank3_res_prob = []

rank1_res_prob_norm = []
rank2_res_prob_norm = []
rank3_res_prob_norm = []

p_sitting_prob = []
p_standing_prob = []
p_walking_prob = []

p_duration_lis =[]

pre_act_list = []
pre_act_symbol_list = []



location_res = []
audio_type_res = []
motion_type_res = []
object_res = []

res_prob_audio_motion = []


transition_motion_occur = []

def get_pre_act_list():

    return []

# init
for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
    res_prob[act] = []

episode_count = 20
batch_size = 256

# stores the reward per episode
scores = deque(maxlen=100)

#w_accuracy = 0.6
#w_energy = 0.3
#w_privacy = 0.1

#w_accuracy = 0.2
#w_energy = 0.6
#w_privacy = 0.8

#w_accuracy = 0.2
#w_energy = 0.1

# can not work well, 0.2 * (-1) + 0.35  = 0.15, 0.35-0.45=-0.15
w_accuracy = 0.3
w_energy = 0.25
w_privacy = 0.45


#w_accuracy = 0.3
#w_energy = 0.5
#w_privacy = 1 - w_accuracy - w_energy
# 1 = w_accuracy + w_energy + w_privacy

time_location_dict = time_adl_res_dict.time_location_dict
time_sound_dict = time_adl_res_dict.time_sound_dict
time_motion_dict = time_adl_res_dict.time_motion_dict
time_object_dict = time_adl_res_dict.time_object_dict

print('len time_location_dict:', len(time_location_dict))
time_exist_dict = time_adl_res_dict.time_exist_dict


import ground_truth_dict_dataset0819
ground_truth_dict = ground_truth_dict_dataset0819.ground_truth_dict

ground_truth_dict = sorted(ground_truth_dict.items(), key=sorter_dict, reverse=False)

cache_ground_truth_dict  = time_adl_res_dict.cache_ground_truth_dict

def get_activity_by_time_str(cur_time_str):

    if cur_time_str in cache_ground_truth_dict.keys():
        return cache_ground_truth_dict[cur_time_str]

    target_folder_time_str = get_target_folder_time_str(cur_time_str)
    if target_folder_time_str == '':
        cache_ground_truth_dict[cur_time_str] = ''
        return ''

    
    key = datetime.strptime(target_folder_time_str, tools_ascc.ASCC_DATASET_DATE_HOUR_TIME_FORMAT_DIR) 
#    print("in get_activity_by_time_str target_folder_time_str:", target_folder_time_str, " time_str:", cur_time_str, ' ', key)
    res = ''
    
    for k, v in ground_truth_dict:
        tmp_start = datetime.strptime(k, tools_ascc.ASCC_DATASET_DATE_HOUR_TIME_FORMAT_DIR) 

#        print("in get_activity_by_time_str target_folder_time_str:", ' k:', k, ' ', tmp_start)
        if key >= tmp_start:
            res = v[2]
        else:
            print("in get_activity_by_time_str target_folder_time_str:", target_folder_time_str, " time_str:", cur_time_str,' res:', res)
            cache_ground_truth_dict[cur_time_str] = res

            return res

    return ''

def get_target_folder_time_str(cur_time_str):
    target_time_str = ''
    
    if cur_time_str in time_exist_dict.keys():
        return time_exist_dict[cur_time_str]

    try:
        image_dir_name = tools_ascc.get_exist_image_dir(cur_time_str)
        # /home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES_V2/ADL_HMM_BAYES/Ascc_Dataset_0819//Image/2009-12-11-08-46-27/
        target_time_str = image_dir_name.split('Image/')[1].rstrip('/')
    except Exception as e:
        print("err:", e)
        target_time_str = ''

    time_exist_dict[cur_time_str] = target_time_str

    return target_time_str

def get_activity_prediction_by_hmm():
    res = {}
    for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
        # hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_list, act, activity_duration)
        hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_symbol_list, act, activity_duration)
        res[act] = hmm_prob

    sd = sorted(res.items(), key=sorter_take_count, reverse=True)

    # return the act with high prob TODO
    for k,v in sd:
        print('res2:', k, ' v:', v)
        return k
    # break

    return ''

time_detected_act_dict =  time_adl_res_dict.time_detected_act_dict

def get_activity_by_action(cur_time_str, action):
    # env.running_time
    # test_time_str = '2009-12-11 12:58:33'
    # cur_time = env.get_running_time()
    # cur_time_str = cur_time.strftime(rl_env_ascc.DATE_HOUR_TIME_FORMAT)
    # print('cur_time:', cur_time)
    
    bayes_model_location.set_time(cur_time_str)
    bayes_model_motion.set_time(cur_time_str)
    bayes_model_audio.set_time(cur_time_str)
    bayes_model_object.set_time(cur_time_str)

    location = ""
    object_dict = ""
    audio_type = ""
    motion_type = ""


    target_folder_time_str = get_target_folder_time_str(cur_time_str)
    key = (target_folder_time_str, action)
    # print("target_folder_time_str:", target_folder_time_str, " time_str:", cur_time_str, ' location:', location)
    if key in time_detected_act_dict.keys():
        return time_detected_act_dict[key] 

    if target_folder_time_str == '':
        return ''

    if target_folder_time_str in time_location_dict.keys():
        location, location_prob = time_location_dict[target_folder_time_str]
        object_dict = time_object_dict[target_folder_time_str]
        audio_type, audio_type_prob = time_sound_dict[target_folder_time_str]
        motion_type, motion_type_prob = time_motion_dict[target_folder_time_str]

        bayes_model_location.set_location_prob(location_prob)
        bayes_model_audio.set_audio_type_prob(float(audio_type_prob))

        print("the dict works, target_folder_time_str:", target_folder_time_str, " time_str:", cur_time_str, ' location:', location)

    else:
        # if action == 1 or action == 2 or action == 5 or action == 6:
        location, location_prob = get_location_by_activity_cnn(cur_time_str)
        print("not works, target_folder_time_str:", target_folder_time_str, " time_str:", cur_time_str, ' location:', location)
        bayes_model_location.set_location_prob(location_prob)

        #location_res.append([location, location_prob])

        
        object_dict = get_object_by_activity_yolo(cur_time_str)

        time_location_dict[target_folder_time_str] = (location, location_prob)
        time_object_dict[target_folder_time_str] = object_dict

    # bayes_model_object.set_object_prob(object_prob)

    # if action == 0 or action == 2 or action == 4 or action == 6:
        audio_type, audio_type_prob = get_audio_type_by_activity_cnn(cur_time_str)
        bayes_model_audio.set_audio_type_prob(float(audio_type_prob))
        #audio_type_res.append([audio_type, audio_type_prob])

        time_sound_dict[target_folder_time_str] = (audio_type, audio_type_prob)



        motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
        bayes_model_motion.set_motion_type_prob(motion_type_prob)
        #motion_type_res.append([motion_type, motion_type_prob])

        time_motion_dict[target_folder_time_str] = (motion_type, motion_type_prob)

    
    # object_res.append([object, object_prob])

    print('location:', location)
    print('object:', object_dict)
    print('audio_type:', audio_type)
    print('motion_type:', motion_type)

    
    
    heap_prob = []
    heap_prob_audio_motion = []

    p2_res_dict = {}

    # RL_ACTION_DICT = {
    # 0: WMU_audio,  
    # 1: WMU_vision, 
    # 2: WMU_fusion,  
    # 3: Robot_audio_vision,
    # 4: Robot_WMU_audio, # robot and WMU both capture data
    # 5: Robot_WMU_vision,
    # 6: Robot_WMU_fusion,
    # 7: Nothing
    # }

    for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
        # hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_list, act, activity_duration)
        hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_symbol_list, act, activity_duration)

        p1 = 1
        if action == 1 or action == 2 or action == 5 or action == 6:
            p1 = bayes_model_location.get_prob(pre_act_list, act, location, 0)

        p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, 0)

        p3 = 1
        if action == 0 or action == 2 or action == 4 or action == 6:
            p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, 0)
        
        p4 =1 

        if location == constants.LOCATION_LIVINGROOM:

            res_object = location
            res_object_p = constants.MIN_Prob
            
            object_laptop_flag = False
            object_book_flag = False
            for object, prob in object_dict:
                if object == constants.OBJECT_LAPTOP:
                    object_laptop_flag = True
                elif object == constants.OBJECT_BOOK:
                    object_book_flag = True


            for object, prob in object_dict:
                print('in living room:', object, ' cur_time_str:', cur_time_str)
                if object == constants.OBJECT_LAPTOP:
                    res_object = object
                    res_object_p = prob
                    bayes_model_object.set_object_prob(res_object_p)
                    p4 = bayes_model_object.get_prob(pre_act_list, act, res_object, activity_duration)
                    break
                elif object == constants.OBJECT_BOOK:

                    if object_laptop_flag == True:
                        continue

                    res_object = object
                    res_object_p = prob
                    bayes_model_object.set_object_prob(res_object_p)
                    p4 = bayes_model_object.get_prob(pre_act_list, act, res_object, activity_duration)


                elif object == constants.OBJECT_TV:
                    if object_book_flag == True or object_laptop_flag == True:
                        continue
                    
                    res_object = object
                    res_object_p = prob
                    bayes_model_object.set_object_prob(res_object_p)
                    p4 = bayes_model_object.get_prob(pre_act_list, act, res_object, activity_duration)

        p = p1*p2*p3*p4 * hmm_prob
            
        #res_prob[act].append(p)
        heap_prob.append((act, p, cur_time_str))

        p2_res_dict[act] = p2
        
    print('heap_prob len:', len(heap_prob))
    top3_prob = sorted(heap_prob, key=sorter_take_count,reverse=True)[:3]
    #print('top3_prob:', top3_prob)

    
    activity_detected = top3_prob[0][0]

    p_activity_end = motion_adl_bayes_model.get_end_of_activity_prob_by_duration(activity_duration, activity_detected)

    #p_duration_lis.append(p_activity_end)


    rank1_res_prob.append(top3_prob[0])
    #rank2_res_prob.append(top3_prob[1])
    #rank3_res_prob.append(top3_prob[2])

    rank1_res_prob_norm.append(p_activity_end)
   # p_rank2 = (1-p_activity_end) * (rank2_res_prob[-1][1] + 1e-200)/(rank2_res_prob[-1][1]+ 1e-200+rank3_res_prob[-1][1]+ 1e-200)
   # rank2_res_prob_norm.append(p_rank2)
   # p_rank3 = (1-p_activity_end) * (rank3_res_prob[-1][1] + 1e-200)/(rank2_res_prob[-1][1]+ 1e-200+rank3_res_prob[-1][1]+ 1e-200)
   # rank3_res_prob_norm.append(p_rank3)
    # print('rank1_res_prob_norm:', rank1_res_prob_norm)
    # print('rank2_res_prob_norm:', rank2_res_prob_norm)
    # print('rank3_res_prob_norm:', rank3_res_prob_norm)



    # pre_activity = top3_prob[0][0]
    cur_activity = top3_prob[0][0]
    # activity_begin_time = cur_time
    time_detected_act_dict[key] = cur_activity

    del heap_prob

    return cur_activity


# init DQN agent
while(pre_activity == ''):
    # open camera

    audio_data, vision_data, motion_data, transition_motion = env.step(rl_env_ascc.WMU_FUSION_ACTION)

    # env.running_time
    # test_time_str = '2009-12-11 12:58:33'
    cur_time = env.get_running_time()
    cur_time_str = cur_time.strftime(rl_env_ascc.DATE_HOUR_TIME_FORMAT)
    print('cur_time:', cur_time)
    
    bayes_model_location.set_time(cur_time_str)
    bayes_model_motion.set_time(cur_time_str)
    bayes_model_audio.set_time(cur_time_str)
    bayes_model_object.set_time(cur_time_str)

    detected_activity = get_activity_by_action(cur_time_str, rl_env_ascc.WMU_FUSION_ACTION)
    if detected_activity == '':
        continue
    
    
    # rank_res.append((detected_activity, '1', cur_time_str))
    
    pre_activity = detected_activity
    cur_activity = detected_activity
    activity_begin_time = cur_time
    
motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
motion_feature = motion_feature_extractor(motion_type) # [0, 0, 0, 0, 0, 1]
battery_feature = [0, 0]
#adl_hidden_feature = [1, 2, 4, 5, 5, 5]  # to be done
predicted_activity = get_activity_prediction_by_hmm()
# adl_hidden_feature = adl_hidden_feature_extractor(predicted_activity) # seems useless
current_activity = get_activity_by_time_str(cur_time_str)
current_activity_duration = (cur_time - activity_begin_time).seconds / 60 # in minutes

current_activity_feature = adl_hidden_feature_extractor(current_activity)
current_activity_duration_feature = current_activity_duration

motion_feature = list(motion_feature)
current_activity_feature = list(current_activity_feature)
current_activity_duration_feature = [current_activity_duration_feature]
state = motion_feature + battery_feature + motion_feature + current_activity_feature + current_activity_duration_feature
state_size = len(state)
state = np.reshape(state, [1, state_size])
print("state_size:", state_size)
print("features:", state)

actions = []

action_space = list(rl_env_ascc.RL_ACTION_DICT.keys())

import rl_ascc_dqn
#agent = rl_ascc_dqn.DQNAgent(state.size, action_space, episodes=500*2.5)


# for test and reload the pretrained model
agent = rl_ascc_dqn.DQNAgent(state.size, action_space, episodes=500*2.5, epsilon = 0.26)
agent.load_weights()



total_wmu_cam_trigger_times = []
total_wmu_mic_trigger_times = []

for episode in range(episode_count):

    location_res = []
    audio_type_res = []
    motion_type_res = []
    object_res = []

    object_dict = {}
    rank_res = []

    pre_act_list = []
    pre_act_symbol_list = []

    pre_activity = ''

    rank1_res_prob = []
    rank2_res_prob = []
    rank3_res_prob = []

    rank1_res_prob_norm = []
    rank2_res_prob_norm = []
    rank3_res_prob_norm = []
    
    transition_occur_cnt = 0
    motion_transition_occur_cnt = 0
    env.reset()
    while(pre_activity == ''):
        # open camera

        audio_data, vision_data, motion_data, transition_motion = env.step(rl_env_ascc.WMU_FUSION_ACTION)

        # env.running_time
        # test_time_str = '2009-12-11 12:58:33'
        cur_time = env.get_running_time()
        cur_time_str = cur_time.strftime(rl_env_ascc.DATE_HOUR_TIME_FORMAT)
        print('cur_time:', cur_time)
        
        bayes_model_location.set_time(cur_time_str)
        bayes_model_motion.set_time(cur_time_str)
        bayes_model_audio.set_time(cur_time_str)
        bayes_model_object.set_time(cur_time_str)

        detected_activity = get_activity_by_action(cur_time_str, rl_env_ascc.WMU_FUSION_ACTION)
        if detected_activity == '':
            continue
        
        if OUTPUT_RANK:
            rank_res.append((detected_activity, '1', cur_time_str))
        
        pre_activity = detected_activity
        cur_activity = detected_activity
        activity_begin_time = cur_time

        pre_act_list.append(pre_activity)

        node = tools_ascc.Activity_Node_Observable(pre_activity, tools_ascc.get_activity_type(cur_time_str), 0)
        pre_activity_symbol = node.activity_res_generation()
        pre_act_symbol_list.append(pre_activity_symbol)
        
    motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
    pre_motion_type = motion_type
    motion_feature = motion_feature_extractor(motion_type) # [0, 0, 0, 0, 0, 1]
    battery_feature = [0, 0]
    #adl_hidden_feature = [1, 2, 4, 5, 5, 5]  # to be done
    predicted_activity = get_activity_prediction_by_hmm()
    # adl_hidden_feature = adl_hidden_feature_extractor(predicted_activity) # seems useless
    current_activity = get_activity_by_time_str(cur_time_str)
    current_activity_duration = (cur_time - activity_begin_time).seconds / 60 # in minutes

    current_activity_feature = adl_hidden_feature_extractor(current_activity)
    current_activity_duration_feature = current_activity_duration

    # features = motion_feature
    # features.extend(battery_feature)
    # features.extend(motion_feature)
    # print("features:", features)
    motion_feature = list(motion_feature)
    current_activity_feature = list(current_activity_feature)
    current_activity_duration_feature = [current_activity_duration_feature]
    state = motion_feature + battery_feature + motion_feature + current_activity_feature + current_activity_duration_feature
    state_size = len(state)
    state = np.reshape(state, [1, state_size])
    print("state_size:", state_size)
    print("features:", state)

    actions = []

    action_space = list(rl_env_ascc.RL_ACTION_DICT.keys())

    # state = env.reset()


    previous_motion_feature = motion_feature

    total_reward = 0


    
    activity_rank_hit_times = 0
    activity_rank_empty_times = 0
    # reinforcement learning part

    rember_cnt = 0
    while(not env.done):
        start_t_iter = timer()

        location = ''
        object = ''
        #motion_type = ''
        audio_type = ''

        living_room_check_flag = False

        object_dict = {}

        # agent chose an action based on the state
        print('Env Running:', cur_time_str, " evn.runing:", env.get_running_time()) 
        action = agent.act(state)
        # print("Env state:", state)
        # print("Env action: ", action, " ", rl_env_ascc.RL_ACTION_DICT[action])

        # env check the action and the cost time
        # action = 6 # rl_env_ascc.Robot_WMU_fusion # to get the time recognition dict
        env.step(action)


        reward_energy = env.get_reward_energy(action)

        ground_truth_activity = get_activity_by_time_str(cur_time_str) # TODO
        detected_activity = ground_truth_activity

        end_t_iter = timer()
        print(" 0 after get activity by time each iter takes:", end_t_iter - start_t_iter)

        reward_privacy = env.get_reward_privacy(action, ground_truth_activity)

        reward_accuracy = 0

        # Todo 
        # detected_activity = get_activity_by_action(cur_time_str, action)
        # print("detected_activity:", detected_activity)
        # print("pre_activity:", pre_activity)
        # cur_time = env.get_running_time()
        # cur_time_str = cur_time.strftime(rl_env_ascc.DATE_HOUR_TIME_FORMAT)

        pre_activity = pre_act_list[-1]


        # walking motion
        motion_transition_occur_flag = False
        if pre_motion_type != motion_type and (pre_motion_type == 'walking' or motion_type == 'walking'):
            print('activity:', pre_activity, ' ', ground_truth_activity)
            motion_transition_occur_cnt += 1
            print('motion_transition_occur_cnt:', motion_transition_occur_cnt)
            motion_transition_occur_flag = True

        if pre_activity != ground_truth_activity and ground_truth_activity!='':
            print("motion_type:", pre_motion_type, " ", motion_type)
            print('activity:', pre_activity, ' ', ground_truth_activity)
            transition_occur_cnt += 1
            #real_transition_occur_cnt += 1
            print('transition_occur_cnt:', transition_occur_cnt)
 

        pre_motion_type = motion_type


        if rl_env_ascc.RL_ACTION_DICT[action] == rl_env_ascc.Robot_audio_vision or rl_env_ascc.RL_ACTION_DICT[action] == rl_env_ascc.Robot_WMU_fusion \
            or rl_env_ascc.RL_ACTION_DICT[action] == rl_env_ascc.Robot_WMU_audio or rl_env_ascc.RL_ACTION_DICT[action] == rl_env_ascc.Robot_WMU_vision \
            or rl_env_ascc.RL_ACTION_DICT[action] == rl_env_ascc.WMU_vision or rl_env_ascc.RL_ACTION_DICT[action] == rl_env_ascc.WMU_fusion:
                
            activity_rank_hit_times += 1
            reward_accuracy = 1
            #if pre_activity != ground_truth_activity and ground_truth_activity!='':
            #    reward_accuracy = 1 * new_activity_factor

        else:
            detected_activity = get_activity_by_action(cur_time_str, action)
            print("detected_activity:", detected_activity)
            print("pre_activity:", pre_activity)

            if detected_activity == '':
                activity_rank_empty_times += 1
                reward_accuracy = 0
            
            elif detected_activity == ground_truth_activity or (BATHROOM in detected_activity and BATHROOM in ground_truth_activity):
                activity_rank_hit_times += 1
                if detected_activity != pre_activity:
                    reward_accuracy = 1 * new_activity_factor
                else:
                    reward_accuracy = 1
            elif ground_truth_activity != '':
                reward_accuracy = -1


        if ground_truth_activity == '':
            activity_rank_empty_times += 1
            reward_accuracy = 0
            reward_energy = 0



        if rl_env_ascc.RL_ACTION_DICT[action] == rl_env_ascc.Nothing:
            reward_accuracy = 0

        if OUTPUT_RANK:
            rank_res.append((detected_activity, '1', cur_time_str))
        
        end_t_iter = timer()
        print(" 1 each iter takes:", end_t_iter - start_t_iter)

        # TODO: in real test, cur_activity = detected_activity
        cur_activity = ground_truth_activity

        if cur_activity != pre_act_list[-1] and cur_activity != '':
            print("pre_activity:", pre_act_list[-1], " cur_activity:", cur_activity)
            print("pre_act_list:", pre_act_list)
            pre_act_list.append(cur_activity)
            node = tools_ascc.Activity_Node_Observable(cur_activity, tools_ascc.get_activity_type(cur_time_str), 0)
            pre_activity_symbol = node.activity_res_generation()
            pre_act_symbol_list.append(pre_activity_symbol)
            # pre_activity = detected_activity
            activity_begin_time = env.get_running_time()
        

        end_t_iter = timer()
        print(" 2 each iter takes:", end_t_iter - start_t_iter)

        reward = reward_accuracy*w_accuracy - reward_energy*w_energy - reward_privacy*w_privacy
        #if motion_transition_occur_flag == True:
        #    reward = reward + MOTION_TRANSITION_REWARD

        print("Env motion:", pre_motion_type)
        print("Env state:", state)
        print("Env action: ", action, " ", rl_env_ascc.RL_ACTION_DICT[action])
        print("Env reward:", reward, " accuracy|energy|privacy:", reward_accuracy, ", ", reward_energy, ", ", reward_privacy)
        print("Env truth activity:", ground_truth_activity)
        print('Env Running (after step):', cur_time_str, " evn.runing:", env.get_running_time()) 

        wmu_mic_times, wmu_cam_times = env.get_wmu_sensor_trigger_times()
        battery_feature = [wmu_mic_times, wmu_cam_times]

        # after the action, time changes and motion changes
        cur_time = env.get_running_time()
        cur_time_str = cur_time.strftime(rl_env_ascc.DATE_HOUR_TIME_FORMAT)
        motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
        if motion_type == '':
            motion_type = pre_motion_type
            print('pre_motion for empty detection:', pre_motion_type)

        next_motion_feature = motion_feature_extractor(motion_type)


        current_activity = get_activity_by_time_str(cur_time_str)
        current_activity_duration = (cur_time - activity_begin_time).seconds / 60 # in minutes

        #end_t_iter = timer()

        current_activity_feature = adl_hidden_feature_extractor(current_activity)
        current_activity_duration_feature = current_activity_duration


        next_motion_feature = list(next_motion_feature)
        current_activity_feature = list(current_activity_feature)
        current_activity_duration_feature = [current_activity_duration_feature]
        next_state = next_motion_feature + battery_feature + previous_motion_feature + current_activity_feature + current_activity_duration_feature
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, env.done)
        rember_cnt = rember_cnt + 1
        
        state = next_state
        total_reward += reward
        previous_motion_feature = next_motion_feature

        end_t_iter = timer()
        print("each iter takes:", end_t_iter - start_t_iter)



        if env.totol_check_times % 500 == 0:
            print("===================================================")
            print("total check times:", env.totol_check_times)
            print("Env wmu mic trigger times:", env.wmu_mic_times)
            print("Env wmu cam trigger times:", env.wmu_cam_times)
            print("Rewards:", total_reward)

        print("===================================================")

        if rember_cnt >= batch_size:
            start_t = timer()
            #agent.replay(batch_size)
            agent.replay2(batch_size)
            end_t = timer()
            print("replay takes:", end_t - start_t, 'replay times:', agent.replay_counter)
            #print("agent replay(len memeory):", len(agent.memory))
            rember_cnt = 0

        if env.done:
            print("episode: {}/{}, episode_reward: {}, e: {:.2}"
            .format(episode, episode_count-1, total_reward, agent.epsilon))

            agent.update_replay_memory()
            print("agent update replay  memeory:", len(agent.memory))

    print(mem_top(limit=15,width=180))
    tools_ascc.show_memory()

    print("time_location_dict:", time_location_dict)
    print("time_object_dict:", time_object_dict)
    print("time_sound_dict:", time_sound_dict)
    print("time_motion_dict:", time_motion_dict)
    print("time_exist_dict:", time_exist_dict)
    print("time_detected_act_dict:", time_detected_act_dict)
    print("cache_ground_truth_dict:", cache_ground_truth_dict)
    print("cache_ground_truth_dict:", len(cache_ground_truth_dict))

   
    # agent.replay(len(agent.memory)/2)


    
    

    
    scores.append(total_reward)
    # plot rewards 
    plotone(scores, "rl_reward.png")
    print("plotone scores:", scores)

    total_wmu_cam_trigger_times.append(env.wmu_cam_times)
    total_wmu_mic_trigger_times.append(env.wmu_mic_times)
    plot(total_wmu_cam_trigger_times, total_wmu_mic_trigger_times, label1="camera", label2="microphone")
    print("total_wmu_cam_trigger_times:", total_wmu_cam_trigger_times)
    print("total_wmu_mic_trigger_times:", total_wmu_mic_trigger_times)
    print("total_privacy_occur_cnt:", env.privacy_occur_cnt)

    print("rank res", len(rank_res))
    print("rank res", rank_res)
    print("hit times:{}, empty times:{}, total times:{}".format(activity_rank_hit_times, activity_rank_empty_times, len(rank_res)))
    print("hit ratio:", activity_rank_hit_times*1.0/(len(rank_res)+1), "hit_empty ratio:", (activity_rank_hit_times + activity_rank_empty_times)*1.0/(len(rank_res)+1))

    print('rank1_res_prob_norm:', rank1_res_prob_norm)
    # print('rank2_res_prob_norm:', rank2_res_prob_norm)
    # print('rank3_res_prob_norm:', rank3_res_prob_norm)
    # plot(total_wmu_cam_trigger_times, "wmu_cam_times.png")
    # plot(total_wmu_mic_trigger_times, "wmu_mic_times.png")
    # while not env.done

    agent.save_weights()
    gc.collect()




print("===================================================")
print("rank res", rank_res)
# end episode for




