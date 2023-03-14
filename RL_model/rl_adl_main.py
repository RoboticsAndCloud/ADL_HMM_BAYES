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


#import constants
#from tkinter.messagebox import NO

import hmm
import rl_env_ascc
import motion_adl_bayes_model
import tools_ascc
import constants

import tools_sql
import matplotlib.pyplot as plt




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

def plot(rewards, figure = "reward.png"):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    # plt.legend()
    plt.savefig(figure)

    # plt.show()
    plt.clf()

def sorter_take_count(elem):
    # print('elem:', elem)
    return elem[1]


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

def get_activity_by_time_str(activity_time_str):

    

    day_begin, day_end = tools_ascc.get_day_begin_end(activity_date_dict,
                                            activity_begin_dict, activity_end_dict)

    hit_activity_check_times = 0

    print("=====================================")
    print("Date:", day_time_str)
    print("activity_begin_dict", len(activity_begin_dict))
    print("activity_end_dict", len(activity_end_dict))

    motion_activity_cnt = 0
    import collections
    output_dict = {}
    output_dict2 = {}  # timestamp is at the beginning

    output_miss_event_dict = {}

    last_hit_time_list = []
    last_miss_time_set = set()
    print("Activity", "\t", "Start", "\t", "End", "\t", "Duration")

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
                print("A end < day begin, ignore:", a_end, day_begin)
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
                    hit_activity_check_times = hit_activity_check_times + 1
                    ## Note: in dict, the time is out off order
                    last_hit_time_list.append(hit_time)
                    # print("#####:hit time:", hit_time, key)
                    return key
       
    return ''





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

    # Mapping
    # should be act : probability
    # /home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/src/
    # ascc_room_activity_test.py
    motion_type_list, prob_list = tools_ascc.get_activity_by_motion_dnn(time_str, action='vision')
    motion_type = ''
    prob = -1
    if len(motion_type_list) > 0:
        motion_type = motion_type_list[-1]
        prob = float(prob_list[-1])


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

    # Applying the function on input class vector
    from keras.utils import to_categorical
    output_matrix = to_categorical(class_vector, num_classes = 6, dtype ="int32")

    print(output_matrix)
    # [[0 0 0 0 0 1]]

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

episode_count = 3
batch_size = 64

# stores the reward per episode
scores = deque(maxlen=100)

w_accuracy = 0.5
w_energy = 0.3
w_privacy = 1 - w_accuracy - w_energy
# 1 = w_accuracy + w_energy + w_privacy


def get_activity_by_action(action):
    # env.running_time
    # test_time_str = '2009-12-11 12:58:33'
    cur_time = env.get_running_time()
    cur_time_str = cur_time.strftime(rl_env_ascc.DATE_HOUR_TIME_FORMAT)
    print('cur_time:', cur_time)
    
    bayes_model_location.set_time(cur_time_str)
    bayes_model_motion.set_time(cur_time_str)
    bayes_model_audio.set_time(cur_time_str)
    bayes_model_object.set_time(cur_time_str)


    location, location_prob = get_location_by_activity_cnn(cur_time_str)
    bayes_model_location.set_location_prob(location_prob)

    object_dict = get_object_by_activity_yolo(cur_time_str)
    # bayes_model_object.set_object_prob(object_prob)

    audio_type, audio_type_prob = get_audio_type_by_activity_cnn(cur_time_str)
    bayes_model_audio.set_audio_type_prob(float(audio_type_prob))

    motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
    bayes_model_motion.set_motion_type_prob(motion_type_prob)
    # bayes_model_motion.set_motion_type(motion_type)

    location_res.append([location, location_prob])
    audio_type_res.append([audio_type, audio_type_prob])
    motion_type_res.append([motion_type, motion_type_prob])
    # object_res.append([object, object_prob])

    print('location:', location)
    print('object:', object_dict)
    print('audio_type:', audio_type)
    print('motion_type:', motion_type)

    
    
    heap_prob = []
    heap_prob_audio_motion = []

    p2_res_dict = {}

    for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
        # hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_list, act, activity_duration)
        hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_symbol_list, act, activity_duration)


        p1 = bayes_model_location.get_prob(pre_act_list, act, location, 0)
        p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, 0)
        p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, 0)
        # p4 = bayes_model_object.get_prob(pre_act_list, act, object, 0)

        if action == 0: # audio
            p1 = 1
        if action == 1:
            p3 = 1
        
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
            
        res_prob[act].append(p)
        heap_prob.append((act, p, cur_time_str))

        p2_res_dict[act] = p2
        
    print('heap_prob:', heap_prob)
    top3_prob = sorted(heap_prob, key=sorter_take_count,reverse=True)[:3]
    print('top3_prob:', top3_prob)

    
    activity_detected = top3_prob[0][0]

    p_activity_end = motion_adl_bayes_model.get_end_of_activity_prob_by_duration(activity_duration, activity_detected)

    p_duration_lis.append(p_activity_end)


    rank1_res_prob.append(top3_prob[0])
    rank2_res_prob.append(top3_prob[1])
    rank3_res_prob.append(top3_prob[2])

    rank1_res_prob_norm.append(p_activity_end)
    p_rank2 = (1-p_activity_end) * (rank2_res_prob[-1][1] + 1e-200)/(rank2_res_prob[-1][1]+ 1e-200+rank3_res_prob[-1][1]+ 1e-200)
    rank2_res_prob_norm.append(p_rank2)
    p_rank3 = (1-p_activity_end) * (rank3_res_prob[-1][1] + 1e-200)/(rank2_res_prob[-1][1]+ 1e-200+rank3_res_prob[-1][1]+ 1e-200)
    rank3_res_prob_norm.append(p_rank3)
    print('rank1_res_prob_norm:', rank1_res_prob_norm)
    print('rank2_res_prob_norm:', rank2_res_prob_norm)
    print('rank3_res_prob_norm:', rank3_res_prob_norm)



    pre_activity = top3_prob[0][0]
    cur_activity = top3_prob[0][0]
    activity_begin_time = cur_time

    pre_act_list.append(pre_activity)

    node = tools_ascc.Activity_Node_Observable(pre_activity, tools_ascc.get_activity_type(cur_time_str), 0)
    pre_activity_symbol = node.activity_res_generation()
    pre_act_symbol_list.append(pre_activity_symbol)

    activity = cur_activity
    time = cur_time_str
    image_source = location + LOCATION_DIR_SPLIT_SYMBOL + image_dir
    sound_source = audio_type
    motion_source = motion_type
    # tools_sql.insert_adl_activity_data(activity, time, image_source, sound_source, motion_source)
    # print('insert int to db: activity:', activity, ' cur_time:', cur_time_str)

    tools_ascc.get_activity_duration_by_date()

    return cur_activity

for episode in range(episode_count):

    object_dict = {}
    while(pre_activity == ''):
        # open camera

        audio_data, vision_data, motion_data, transition_motion = env.step(rl_env_ascc.FUSION_ACTION)  # rl_env_ascc.FUSION_ACTION?

        # env.running_time
        # test_time_str = '2009-12-11 12:58:33'
        cur_time = env.get_running_time()
        cur_time_str = cur_time.strftime(rl_env_ascc.DATE_HOUR_TIME_FORMAT)
        print('cur_time:', cur_time)
        
        bayes_model_location.set_time(cur_time_str)
        bayes_model_motion.set_time(cur_time_str)
        bayes_model_audio.set_time(cur_time_str)
        bayes_model_object.set_time(cur_time_str)

        # todo change to str

        # cur_activity, cur_beginning_activity, cur_end_activity = \
        #     bayes_model_location.get_activity_from_dataset_by_time(cur_time_str)

        # print('cur_time:', cur_time, ' cur_activity:', cur_activity)
        # exit(0)

        """
        2009-12-11 08:42:03.000082	M021	ON	Sleep end
        2009-12-11 08:42:04.000066	M028	ON
        2009-12-11 08:42:06.000089	M020	ON
        """
        # if cur_activity == None or cur_activity == '':
        #     continue


        # detect activity, cur_activity, pre_activity
        # Bayes model
        # location = get_location_by_activity(cur_activity)
        # object = get_object_by_activity(cur_activity)
        # audio_type = get_audio_type_by_activity(cur_activity)
        # motion_type = get_motion_type_by_activity(cur_activity)

        location, location_prob, image_dir = get_location_by_activity_cnn(cur_time_str)
        bayes_model_location.set_location_prob(location_prob)

        object_dict = get_object_by_activity_yolo(cur_time_str)
        # bayes_model_object.set_object_prob(object_prob)

        audio_type, audio_type_prob = get_audio_type_by_activity_cnn(cur_time_str)
        bayes_model_audio.set_audio_type_prob(float(audio_type_prob))

        motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
        bayes_model_motion.set_motion_type_prob(motion_type_prob)
        # bayes_model_motion.set_motion_type(motion_type)

        location_res.append([location, location_prob])
        audio_type_res.append([audio_type, audio_type_prob])
        motion_type_res.append([motion_type, motion_type_prob])
        # object_res.append([object, object_prob])

        print('location:', location)
        print('object:', object_dict)
        print('audio_type:', audio_type)
        print('motion_type:', motion_type)

        
        
        heap_prob = []
        heap_prob_audio_motion = []

        p2_res_dict = {}

        for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
            # hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_list, act, activity_duration)
            hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_symbol_list, act, activity_duration)


            p1 = bayes_model_location.get_prob(pre_act_list, act, location, 0)
            
            #  bayes_model_location.prob_of_location_under_act(location, act) \
                #  * motion_adl_bayes_model.HMM_START_MATRIX[act] /(bayes_model_location.prob_of_location_under_all_acts(location)) * bayes_model_location.prob_of_location_using_vision(location, act)
            
            p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, 0)
            p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, 0)
            # p4 = bayes_model_object.get_prob(pre_act_list, act, object, 0)

            p4 = 1
            p3 = 1

            p = p1*p2*p3*p4 * hmm_prob
                
            res_prob[act].append(p)
            heap_prob.append((act, p, cur_time_str))

            p2_res_dict[act] = p2
            
        print('heap_prob:', heap_prob)
        top3_prob = sorted(heap_prob, key=sorter_take_count,reverse=True)[:3]
        print('top3_prob:', top3_prob)

        
        activity_detected = top3_prob[0][0]
        
        # p_sitting_prob.append(0)
        # p_standing_prob.append(0)
        # p_walking_prob.append(0)
        # if motion_type == motion_adl_bayes_model.MOTION_TYPE_SITTING:
        #     p_sitting_prob[len(p_sitting_prob)-1] = p2
        # elif motion_type == motion_adl_bayes_model.MOTION_TYPE_STANDING:
        #     p_standing_prob[len(p_standing_prob)-1] = p2
        # elif motion_type == motion_adl_bayes_model.MOTION_TYPE_WALKING:
        #     p_walking_prob[len(p_walking_prob)-1] = p2


        
        p_activity_end = motion_adl_bayes_model.get_end_of_activity_prob_by_duration(activity_duration, activity_detected)

        p_duration_lis.append(p_activity_end)


        rank1_res_prob.append(top3_prob[0])
        rank2_res_prob.append(top3_prob[1])
        rank3_res_prob.append(top3_prob[2])

        rank1_res_prob_norm.append(p_activity_end)
        p_rank2 = (1-p_activity_end) * (rank2_res_prob[-1][1] + 1e-200)/(rank2_res_prob[-1][1]+ 1e-200+rank3_res_prob[-1][1]+ 1e-200)
        rank2_res_prob_norm.append(p_rank2)
        p_rank3 = (1-p_activity_end) * (rank3_res_prob[-1][1] + 1e-200)/(rank2_res_prob[-1][1]+ 1e-200+rank3_res_prob[-1][1]+ 1e-200)
        rank3_res_prob_norm.append(p_rank3)
        print('rank1_res_prob_norm:', rank1_res_prob_norm)
        print('rank2_res_prob_norm:', rank2_res_prob_norm)
        print('rank3_res_prob_norm:', rank3_res_prob_norm)



        pre_activity = top3_prob[0][0]
        cur_activity = top3_prob[0][0]
        activity_begin_time = cur_time

        pre_act_list.append(pre_activity)

        node = tools_ascc.Activity_Node_Observable(pre_activity, tools_ascc.get_activity_type(cur_time_str), 0)
        pre_activity_symbol = node.activity_res_generation()
        pre_act_symbol_list.append(pre_activity_symbol)

        activity = cur_activity
        time = cur_time_str
        image_source = location + LOCATION_DIR_SPLIT_SYMBOL + image_dir
        sound_source = audio_type
        motion_source = motion_type
        # tools_sql.insert_adl_activity_data(activity, time, image_source, sound_source, motion_source)
        # print('insert int to db: activity:', activity, ' cur_time:', cur_time_str)


        # TODO top3 data
        # TODO how to get the accuracy

    need_recollect_data = False
    p_check_level = 4
    start_check_interval = 0
    p_less_than_threshold_check_cnt = 0
    start_check_interval_time = None
    living_room_check_times = LIVING_ROOM_CHECK_TIMES_MAX 




    motion_feature = motion_feature_extractor(motion_type) # [0, 0, 0, 0, 0, 1]
    battery_feature = [0, 0]
    adl_hidden_feature = [1, 2, 4, 5, 5, 5]

    features = motion_feature
    features.extend(battery_feature)
    features.extend(motion_feature)
    print("features:", features)

    actions = []

    import rl_ascc_dqn
    agent = rl_ascc_dqn.DQNAgent(len(features), len(actions))

    # state = env.reset()

    # motion, battery, previous_motion
    cur_motion_feature = motion_feature
    previous_motion_feature = motion_feature
    state = cur_motion_feature + battery_feature + previous_motion_feature

    total_reward = 0

    rank_res = []
    # reinforcement learning part
    while(not env.done):

        # TODO:
        # p = rank1_res_prob[-1]
        # if p of previous detection is smaller than threshodl, env.step(rl_env_ascc.FUSION_ACTION)

        location = ''
        object = ''
        motion_type = ''
        audio_type = ''

        living_room_check_flag = False

        object_dict = {}

        # agent chose an action based on the state
        action = agent.act(state)

        # env check the action and the cost time
        env.step(action)

        # TODO: calulate the reward based on accuracy, privacy, energy
        reward_energy = env.get_reward_energy(action)
        reward_privacy = env.get_reward_privacy(action, cur_time_str)

        detected_activity = get_activity_by_action(action)
        ground_truth_activity = get_activity_by_time_str(cur_time_str) # TODO

        rank_res.append((detected_activity, '1', cur_time_str))
        
        reward_accuracy = 0

        if detected_activity == ground_truth_activity:
            if detected_activity != pre_activity:
                reward_accuracy = 1
            else:
                reward_accuracy = 0
        elif ground_truth_activity == '':
            reward_accuracy = 0
        else:
            reward_accuracy = -1

        pre_activity = ground_truth_activity

        if rl_env_ascc.RL_ACTION_DICT[action] == rl_env_ascc.Robot_audio_vision or rl_env_ascc.RL_ACTION_DICT[action] == rl_env_ascc.Robot_WMU_fusion:
            reward_accuracy = 1

        if rl_env_ascc.RL_ACTION_DICT[action] == rl_env_ascc.Nothing:
            reward_accuracy = 0

        
        

        reward = reward_accuracy*w_accuracy - reward_energy*w_energy - reward_privacy*w_privacy

        wmu_mic_times, wmu_cam_times = env.get_wmu_sensor_trigger_times()
        battery_feature = [wmu_mic_times, wmu_cam_times]

        # after the action, time changes and motion changes
        cur_time = env.get_running_time()
        cur_time_str = cur_time.strftime(rl_env_ascc.DATE_HOUR_TIME_FORMAT)
        motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
        next_motion_feature = motion_feature_extractor(motion_type)
        
        next_state = next_motion_feature + battery_feature + previous_motion_feature

        agent.remember(state, action, reward, next_state, env.done)
        state = next_state
        total_reward += reward
        previous_motion_feature = next_motion_feature

        if env.done:
            print("episode: {}/{}, episode_reward: {}, e: {:.2}"
            .format(episode, episode_count-1, total_reward, agent.epsilon))



        


        # if need_recollect_data:
        #     audio_data, vision_data, motion_data, transition_motion = env.step(rl_env_ascc.FUSION_ACTION)  
        #     location, location_prob, image_dir = get_location_by_activity_cnn(cur_time_str)
        #     bayes_model_location.set_location_prob(location_prob)

        #     object_dict = get_object_by_activity_yolo(cur_time_str)
        #     # bayes_model_object.set_object_prob(object_prob)
        #     object = object_dict

        #     audio_type, audio_type_prob = get_audio_type_by_activity_cnn(cur_time_str)
        #     bayes_model_audio.set_audio_type_prob(audio_type_prob)

        #     motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
        #     bayes_model_motion.set_motion_type_prob(motion_type_prob)
        #     # bayes_model_motion.set_motion_type(motion_type)


        #     location_res.append([location, location_prob])
        #     audio_type_res.append([audio_type, audio_type_prob])
        #     # motion_type_res.append([motion_type, motion_type_prob])

        #     if location == '':
        #         print('Location  empty:', cur_time)
        #         cur_time = env.get_running_time()
        #         cur_time_str = cur_time.strftime(rl_env_ascc.DATE_HOUR_TIME_FORMAT)
        #         continue

        # else:
        #     # INTERVAL_FOR_COLLECTING_DATA
        #     audio_data, vision_data, motion_data, transition_motion = env.step(rl_env_ascc.MOTION_ACTION)  

        #     # detect transition: the end of walk activity
        #     motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
        #     bayes_model_motion.set_motion_type_prob(motion_type_prob)

        #     location_res.append(['', ''])
        #     audio_type_res.append(['', ''])
        #     # motion_type_res.append([motion_type, motion_type_prob])





        heap_prob = []
        heap_prob_audio_motion = []
        # if transition_motion:
        #     cur_time = env.get_running_time()
        #     cur_time_str = cur_time.strftime(rl_env_ascc.DATE_HOUR_TIME_FORMAT)
        #     print('Env Running:', cur_time) 
            
        #     bayes_model_location.set_time(cur_time_str)
        #     bayes_model_motion.set_time(cur_time_str)
        #     bayes_model_audio.set_time(cur_time_str)
        #     bayes_model_object.set_time(cur_time_str)

        #     cur_activity, cur_beginning_activity, cur_end_activity = \
        #         bayes_model_location.get_activity_from_dataset_by_time(cur_time_str)
                
        #     if cur_activity == None or cur_activity == '' or cur_activity == 'Sleep':
        #         continue

        #     location = get_location_by_activity(cur_activity)
        #     object = get_object_by_activity(cur_activity)
        #     audio_type = get_audio_type_by_activity(cur_activity)
        #     motion_type = get_motion_type_by_activity(cur_activity)

        #     print('location:', location)
        #     print('object:', object)
        #     print('audio_type:', audio_type)
        #     print('motion_type:', motion_type)

        #     # if pre_activity != cur_activity:
        #     #     activity_begin_time = cur_time
            
        #     activity_duration = (cur_time - activity_begin_time).seconds / 60 # in minutes


        #     for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
        #         print("transition step act:", act)
        #         hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_list, act, activity_duration)

        #         p1 = bayes_model_location.get_prob(pre_act_list, act, location, activity_duration)
        #         p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, activity_duration)
        #         p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, activity_duration)
        #         p4 = bayes_model_object.get_prob(pre_act_list, act, object, activity_duration)

        #         p = p1*p2*p3*p4 * hmm_prob
                
        #         print("transition step act:", act)
        #         print('p1:', p1)
        #         print('p2:', p2)
        #         print('p3:', p3)
        #         print('p4:', p4)
        #         print('p:', p)
        #         print("======================================================")

        #         res_prob[act].append(p) 
        #         heap_prob.append((act, p, cur_time_str))

        # else:
        #     # motion data  or audio data
        #     # new activity
        #     # Bayes model prob    

        cur_time = env.get_running_time()
        cur_time_str = cur_time.strftime(rl_env_ascc.DATE_HOUR_TIME_FORMAT)
        print('Env Running:', cur_time) 

        

        bayes_model_location.set_time(cur_time_str)
        bayes_model_motion.set_time(cur_time_str)
        bayes_model_audio.set_time(cur_time_str)
        # bayes_model_object.set_time(cur_time_str)

        # cur_activity, cur_beginning_activity, cur_end_activity = \
        #     bayes_model_location.get_activity_from_dataset_by_time(cur_time_str)

        # if cur_activity == None or cur_activity == '' or cur_activity == 'Sleep':
        #     continue
        
        activity_duration = (cur_time - activity_begin_time).seconds / 60 # in minutes
        # prob_of_activity_by_duration = get_end_of_activity_prob_by_duration(activity_duration, cur_activity)

        # location = get_location_by_activity(cur_activity)
        # object = get_object_by_activity(cur_activity)
        # motion_type = get_motion_type_by_activity(cur_activity)
        # audio_type = get_audio_type_by_activity(cur_activity)

        print('location:', location)
        print('object:', object)
        print('audio_type:', audio_type)
        print('motion_type:', motion_type)

        # # TODO: get the probability of motion type from motion_type = get_motion_type_by_activity(cur_activity)
        # p_sitting_prob.append(0)
        # p_standing_prob.append(0)
        # p_walking_prob.append(0)
        
        # if motion_type == motion_adl_bayes_model.MOTION_TYPE_SITTING:
        #     p_sitting_prob[len(p_sitting_prob)-1] = p2
        # elif motion_type == motion_adl_bayes_model.MOTION_TYPE_STANDING:
        #     p_standing_prob[len(p_standing_prob)-1] = p2
        # elif motion_type == motion_adl_bayes_model.MOTION_TYPE_WALKING:
        #     p_walking_prob[len(p_walking_prob)-1] = p2



        for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
            print("motion step act:", act)
            # hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_list, act, activity_duration)

            hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_symbol_list, act, activity_duration)


            print('hmm_prob:', hmm_prob)

            if need_recollect_data:
                p1 = bayes_model_location.get_prob(pre_act_list, act, location, activity_duration)
                p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, activity_duration)
                p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, activity_duration)
                p4 = bayes_model_object.get_prob(pre_act_list, act, object, activity_duration)

                # p4 = 1
                # p3 = 1
                
                p_audio_motion = p2 * p3 * hmm_prob

                # todo, in the living room, we can collect audio data more times (3 -r times), to confirm the activity
                # or, we can get object activity
                # object == constants.OBJECT_BOOK
                #if audio_type == constants.AUDIO_TYPE_ENV:
            


                if location == constants.LOCATION_LIVINGROOM:
                    living_room_check_flag = True

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

                    # p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, activity_duration)
                    
                p3 = p3 * AUDIO_WEIGHT

                p1 = 1
                p4 = 1
                    
                
                p = p1*p2*p3*p4 * hmm_prob

                
                print("need_recollect_data step act:", act)
                print('hmm_prob:', hmm_prob)
                print('p1_location:', p1)
                print('p2_motion:', p2)
                print('p3_sound:', p3)
                print('p4_object:', p4)
            else:
                p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, activity_duration)
                p = p2 * hmm_prob

                p_audio_motion = p

                
            print("motion step act:", act)
            print('p:', p)
            print("======================================================")

            res_prob[act].append(p) 
            heap_prob.append((act, p, cur_time_str))

            heap_prob_audio_motion.append((act, p_audio_motion, cur_time_str))




        top3_prob = sorted(heap_prob, key=sorter_take_count,reverse=True)[:3]
        activity_detected = top3_prob[0][0]

        top3_prob_audio_motion = sorted(heap_prob_audio_motion, key=sorter_take_count,reverse=True)[:3]


        # # TODO: get the probability of motion type from motion_type = get_motion_type_by_activity(cur_activity)
        # p_sitting_prob.append(0)
        # p_standing_prob.append(0)
        # p_walking_prob.append(0)
        
        # if motion_type == motion_adl_bayes_model.MOTION_TYPE_SITTING:
        #     p_sitting_prob[len(p_sitting_prob)-1] = p2
        # elif motion_type == motion_adl_bayes_model.MOTION_TYPE_STANDING:
        #     p_standing_prob[len(p_standing_prob)-1] = p2
        # elif motion_type == motion_adl_bayes_model.MOTION_TYPE_WALKING:
        #     p_walking_prob[len(p_walking_prob)-1] = p2


    


        print('pre_act_list:', pre_act_list)
        print('pre_act_symbol_list:', pre_act_symbol_list)
        print('heap_prob:', heap_prob)
        top3_prob = sorted(heap_prob, key=sorter_take_count,reverse=True)[:3]
        print('#########top3_prob:', top3_prob)
        # TODO: normalization for the top3 prob
        # if top3_prob[0] < threshold:
        #     need_recollect_data = True

        rank1_res_prob.append(top3_prob[0])
        rank2_res_prob.append(top3_prob[1])
        rank3_res_prob.append(top3_prob[2])

        # tmp_r0 = top3_prob[0][1]
        # tmp_r1 = top3_prob[1][1]
        # tmp_r2 = top3_prob[2][1]

        # tmp_p0_norm = 1.0 * (tmp_r0 + 1e-200) /(tmp_r0 + tmp_r1 + tmp_r2 + 3* 1e-200)
        # tmp_p1_norm = 1.0 * (tmp_r1+ 1e-200) /(tmp_r0 + tmp_r1 + tmp_r2 + 3* 1e-200)
        # tmp_p2_norm = 1.0 * (tmp_r2 + 1e-200)/(tmp_r0 + tmp_r1 + tmp_r2 + 3* 1e-200)

        # rank1_res_prob_norm.append(tmp_r0)
        # rank2_res_prob_norm.append(tmp_p1_norm)
        # rank3_res_prob_norm.append(tmp_p2_norm)
        
        # After got the new activity, We use the duration of the activity to predict the probability
        # Another way is to use the prob in top3 list and calculate the probability, but the probability of top1 is always 1, for example:
        # #########top3_prob: [('Kitchen_Activity', 0.0040946136848312, '2009-12-11 12:19:39'), ('Read', 6.066726746069896e-35, '2009-12-11 12:19:39'), ('Guest_Bathroom', 1.8050356738103724e-35, '2009-12-11 12:19:39')]
        rank1_res_prob_norm.append(p_activity_end)
        p_rank2 = (1-p_activity_end) * (rank2_res_prob[-1][1] + 1e-200)/(rank2_res_prob[-1][1]+ 1e-200+rank3_res_prob[-1][1]+ 1e-200)
        rank2_res_prob_norm.append(p_rank2)
        p_rank3 = (1-p_activity_end) * (rank3_res_prob[-1][1] + 1e-200)/(rank2_res_prob[-1][1]+ 1e-200+rank3_res_prob[-1][1]+ 1e-200)
        rank3_res_prob_norm.append(p_rank3)


        print('#########top3_prob_audio_motion:', top3_prob_audio_motion)
        tmp_r0 = top3_prob_audio_motion[0][1]
        tmp_r1 = top3_prob_audio_motion[1][1]
        tmp_r2 = top3_prob_audio_motion[2][1]

        tmp_p0_norm = 1.0 * (tmp_r0 + 1e-200) /(tmp_r0 + tmp_r1 + tmp_r2 + 3* 1e-200)
        tmp_p1_norm = 1.0 * (tmp_r1+ 1e-200) /(tmp_r0 + tmp_r1 + tmp_r2 + 3* 1e-200)
        tmp_p2_norm = 1.0 * (tmp_r2 + 1e-200)/(tmp_r0 + tmp_r1 + tmp_r2 + 3* 1e-200)
        print('#########top3_prob_audio_motion: r1, r2, r3:', tmp_p0_norm, " ", tmp_p1_norm, " ", tmp_p2_norm)

        if need_recollect_data == True:
            if top3_prob[0][0] != top3_prob_audio_motion[0][0]:
                print('########audio motion vs audio vision motion not equal')
            else:
                print('########audio motion vs audio vision motion equal')


        need_recollect_data = False
        
        p_activity_end = motion_adl_bayes_model.get_end_of_activity_prob_by_duration(activity_duration, activity_detected)

        p_duration_lis.append(p_activity_end)



        if activity_detected == pre_activity:
        # if activity_detected == activity_detected:

            print('p_activity_end:', p_activity_end)
        #     # todo if p_activity_end < 0.2, audio,vision+motion
        #     if (p_activity_end < 0.4) and (p_check_level == 4):
        #         start_check_interval_time = cur_time
        #         need_recollect_data = True
        #         p_check_level = p_check_level -1
        #     if (p_activity_end < 0.3) and (p_check_level == 3):
        #         start_check_interval_time = cur_time
        #         need_recollect_data = True
        #         p_check_level = p_check_level -1
        #     if (p_activity_end < 0.2) and (p_check_level == 2):
        #         start_check_interval_time = cur_time
        #         p_check_level = p_check_level -1

            if p_activity_end < 0.9:    
                if start_check_interval_time == None:
                    start_check_interval_time = cur_time

                start_check_interval = (cur_time - start_check_interval_time).seconds 
                print('start_check_interval:', start_check_interval, ' start_check_interval_time:', start_check_interval_time)

                if (int(start_check_interval) / UNCERTAIN_CHECK_INTERVAL) >= 1:
                    need_recollect_data = True
                    print('reset start_check_interval:', start_check_interval, ' start_check_interval_time:', start_check_interval_time)
                    start_check_interval = 0
                    start_check_interval_time = cur_time

        #     #     need_recollect_data = True
        #     #     p_check_level = p_check_level -1
        #     # if (p_activity_end < 0.1) and (p_check_level == 1):
        #     #     need_recollect_data = True
        #     #     p_check_level = p_check_level -1
        #     # if (p_activity_end < 0.05) and (p_check_level == 0):
        #     #     need_recollect_data = True
        #     #     p_check_level = p_check_level -1
        #     # if (p_activity_end < 0.01):
        #     #     need_recollect_data = True
        #     #     p_check_level = p_check_level -1
        #     print("############need_recollect_data p_check_level:", need_recollect_data, ' ', p_check_level)
        #     if need_recollect_data:
        #         p_less_than_threshold_check_cnt = p_less_than_threshold_check_cnt + 1



        # todo, if rank1 - rank2 < 0.001, p= p+ p*p_audio, to get a more accurate res
        #        
        cur_activity = top3_prob[0][0]

        if location == constants.LOCATION_LOBBY:
            cur_activity = pre_activity
            print('++++++++++++++Around lobby,', cur_time_str)

        # if living_room_check_times == MAX:
        # 

        # incase the wrong prediction from HMM model
        if location == '' and cur_activity != pre_activity:
            cur_activity = pre_activity
            need_recollect_data = True

        if pre_activity != cur_activity:


            if location == constants.LOCATION_LIVINGROOM and living_room_check_times > 0:
                cur_activity = pre_activity
                print('++++++++++++++ in living room, checks:', living_room_check_times, ' ', cur_time_str)

            if new_activity_check_times == DOUBLE_CHECK:
                cur_activity = pre_activity
                new_activity_check_times = new_activity_check_times - 1
                need_recollect_data = TRUE
            else:
                new_activity_check_times = DOUBLE_CHECK

        if pre_activity != cur_activity:
            

            pre_activity = cur_activity
            
            
            activity = cur_activity
            time = cur_time_str
            image_source = location + LOCATION_DIR_SPLIT_SYMBOL + image_dir
            sound_source = audio_type
            motion_source = motion_type
            object_source = ''

            if location == constants.LOCATION_LIVINGROOM:
                for object, prob in object_dict:
                    object_source = object_source + '_' + object



            # tools_sql.insert_adl_activity_data(activity, time, image_source, sound_source, motion_source, object_source)
            # print('insert int to db: activity:', activity, ' cur_time:', cur_time_str)

            pre_act_list.append(pre_activity)

            node = tools_ascc.Activity_Node_Observable(pre_activity, tools_ascc.get_activity_type(cur_time_str), 0)
            pre_activity_symbol = node.activity_res_generation()
            pre_act_symbol_list.append(pre_activity_symbol)

            activity_begin_time = cur_time
            need_recollect_data = False
            p_check_level = 4
            start_check_interval_time = None
            start_check_interval = 0

            # when new activity occrs, double check that, the model will generate the new next activity, we need to confirm again
            # need_recollect_data = True


        
        if len(rank1_res_prob) % 1000 == 0:
            print("===================================================")
            # print out results
            print('rank1:')
            print(rank1_res_prob)

            print('rank2:')
            print(rank2_res_prob)

            print('rank3:')
            print(rank3_res_prob)

            print('res_prob:')
            print(res_prob)

        print('last motion type:', motion_type_res[-1], ' cur motion_type:', motion_type)
        if (motion_type_res[-1][0] == motion_adl_bayes_model.MOTION_TYPE_WALKING) and (motion_type != motion_adl_bayes_model.MOTION_TYPE_WALKING):
            print('Transition occur:', 'last motion type:', motion_type_res[-1], ' cur motion_type:', motion_type)
            transition_motion = TRUE
            need_recollect_data = True
            transition_motion_occur.append(cur_time_str)

            # activity = cur_activity
            # time = cur_time_str
            # image_source = location
            # sound_source = audio_type
            # motion_source = motion_type
            # object_source = ''
            # if location == constants.LOCATION_LIVINGROOM:
            #     for k in object.keys():
            #         object_source = object_source + '_' + k

            # tools_sql.insert_adl_activity_data(activity, time, image_source, sound_source, motion_source, object_source)
            # print('insert int to db: activity:', activity, ' cur_time:', cur_time_str)

        if living_room_check_flag == True and living_room_check_times > 0:
            print('living_room_check_flag occur:', 'times:', living_room_check_flag, ' cur motion_type:', living_room_check_times)
            living_room_check_times = living_room_check_times -1
            need_recollect_data = True
        else:
            living_room_check_times = LIVING_ROOM_CHECK_TIMES_MAX
            living_room_check_flag = False


        location_res.append([location, location_prob])
        audio_type_res.append([audio_type, audio_type_prob])
        motion_type_res.append([motion_type, motion_type_prob])

    if len(agent.memory) >= batch_size:
        agent.replay(batch_size)
    
        scores.append(total_reward)

        # plot rewards 
        plot(scores, "rl_reward.png")

    # while not env.done
print("rank res", rank_res)
# end episode for


print("===================================================")
# print out results
print('rank1:', len(rank1_res_prob))
print(rank1_res_prob)

print('rank2:', len(rank2_res_prob))
print(rank2_res_prob)

print('rank3:', len(rank3_res_prob))
print(rank3_res_prob)

print('res_prob:', len(res_prob))
print(res_prob)

print('p_duration_lis:', len(p_duration_lis))
print(p_duration_lis)

print('rank1_res_prob_norm:', rank1_res_prob_norm)
print('rank2_res_prob_norm:', rank2_res_prob_norm)
print('rank3_res_prob_norm:', rank3_res_prob_norm)

print('location_res len:', len(location_res))
print('location_res:', location_res)
print('audio_type_res:', audio_type_res)
print('motion_type_res:', motion_type_res)

print('transition_motion_occur len:', len(transition_motion_occur))
print('transition_motion_occur:', transition_motion_occur)
print('p_less_than_threshold_check_cnt(uncertain when p < 0.4, 0.3, 0.2):', p_less_than_threshold_check_cnt)

# # motion probabilities during activities
# print('p_sitting_prob:', len(p_sitting_prob))
# print(p_sitting_prob)

# print('p_standing_prob:', len(p_standing_prob))
# print(p_standing_prob)

# print('p_walking_prob:', len(p_walking_prob))
# print(p_walking_prob)

# todo: probability of each activities obtained from the p_duration, for example, cur_activity is 'Read', P_duration(Read) = 0.8, then p(rank2) + p(rank3) = 1-p(Read)=1- 0.8  = 0.2



print("===================================================")

if env.done:
    print("Activity_none_times:", env.activity_none_times)
    print("Expected_activity_none_times:", env.expected_activity_none_times)
    print("Hit times:", env.done_hit_event_times)
    print("Miss times:", env.done_missed_event_times)
    print("Random Miss times:", env.done_random_miss_event_times)
    print("Middle times:", env.done_middle_event_times)
    print("Penalty times:", env.done_penalty_times)
    print("Uncertain times:", env.done_uncertain_times)
    print("Total times:", env.done_totol_check_times)
    print("Residual power:", env.done_residual_power)
    print("Beginning event times:", env.done_beginning_event_times)
    print("Endding event times:", env.done_end_event_times)
    print("Middle event times:", env.done_middle_event_times)
    print("Day End Running time:", env.done_running_time)
    print("Reward:", env.done_reward)
    print("Done status:", env.done)
    print("Sensors Energy cost:", env.done_energy_cost)
    print("Sensors Time cost:", env.done_time_cost)
    print("Sensors Energy total  cost:", env.energy_cost)
    print("Sensors Time total cost:", env.time_cost)
    print("Total Time cost:", env.done_total_time_cost)
    print("Motion_triggered_times:", env.motion_triggered_times)
    print("Hit_activity_check_times", env.hit_activity_check_times)
    end_time_of_wmu = datetime.strptime(env.done_running_time.strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
    print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)

end_time_of_wmu = datetime.strptime(env.done_running_time.strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)


print("Display information:")
# env.display_action_counter()
# env.display_info()
print("===================================================")

print("Activity_none_times \t Expected_activity_none_times \t Hit times \t Miss times \
    \t Random Miss times \t Penalty times \t Uncertain times \t Total times \
    \t Residual power \t Beginning event times \t Endding event times \t Middle event times \
    \t Day End Running time \t Done status \t Duration of Day \t DoneReward \t Reward  \
    \t Sensors energy cost \t Sensors time cost \t total time cost \t Motion_triggered_times \t Hit_activity_check_times \t motion_activity_cnt \t")

print(env.activity_none_times, '\t', env.expected_activity_none_times, '\t',env.done_hit_event_times, '\t', env.done_missed_event_times, \
    '\t', env.done_random_miss_event_times, '\t', env.done_penalty_times, '\t', env.done_uncertain_times, '\t', env.done_totol_check_times, \
    '\t', env.done_residual_power, '\t', env.done_beginning_event_times, '\t', env.done_end_event_times, '\t', env.done_middle_event_times, \
    '\t', env.done_running_time, '\t', env.done, '\t', (end_time_of_wmu - env.day_begin).seconds/3600.0, '\t', 0, '\t', 0, \
    '\t', env.done_energy_cost, '\t', env.done_time_cost, "\t", env.done_total_time_cost, "\t", env.motion_triggered_times, '\t', env.hit_activity_check_times, '\t',env.motion_activity_cnt)

print("===================================================")
