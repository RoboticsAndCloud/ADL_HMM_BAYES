"""
Brief: Bayes model for activity probability, including vision, motion, audio-based model.
Author: Frank
Date: 07/10/2022
"""

from datetime import datetime
from glob import glob
from pickle import TRUE
import random
from re import T

#import constants
#from tkinter.messagebox import NO

import hmm
import real_time_env_ascc
import motion_adl_bayes_model
import tools_ascc
import constants
import threading
from timeit import default_timer as timer

import tools_sql





MILAN_BASE_DATE = '2009-10-16'

MILAN_BASE_DATE_HOUR = '2009-10-16 06:00:00'

TEST_BASE_DATE = '2009-12-11'
DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
HOUR_TIME_FORMAT = "%H:%M:%S"
DAY_FORMAT_STR = '%Y-%m-%d'
FOLDER_DATE_TIME_FORMAT = '%Y%m%d%H%M%S'


UNCERTAIN_CHECK_INTERVAL = 60 # Seconds

LIVING_ROOM_CHECK_TIMES_MAX = 2

DOUBLE_CHECK = 2

LOCATION_DIR_SPLIT_SYMBOL = ':'

AUDIO_WEIGHT = 0.6

g_image_recognition_flag = False
g_sound_recognition_flag = False
g_motion_recognition_flag = False

g_image_recognition_file = ''
g_image_recognition_time = ''
g_sound_recognition_file = ''
g_sound_recognition_time = ''
g_motion_recognition_file = ''
g_motion_recognition_time = ''

g_image_object_recognition_flag = False
g_image_object_recognition_file = ''
g_image_object_recognition_time = ''

g_image_data_location = ''
g_motion_data_location = ''
g_audio_data_location = ''

g_stop = False

CHECK_AND_WAIT_THRESHOLD = 10


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
def get_object_by_activity_yolo(res_file, time_str):


    # get the results
    res_str = tools_ascc.read_res_from_file(res_file)

    """
    class_names=['bathroom','bedroom', 'morning_med', 'reading', 'kitchen','livingroom', 'chores', 'desk_activity', 'dining_room_activity',
                 'eve_med', 'leaving_home', 'meditate']
    """
    # use res_str[0]
    # todo: how to get the result from 10 images, for example, in the dir 08-45-46
    # todo: how to check the motion
    # todo: meditate should be treat as the bedroom activity
    # todo: re-train the model, use more imges
    # todo: check the code, how to get expected activity,  Miss activity: Expected: Sleep ,Detect: Meditate Running time: 2009-12-11 08:45:26
    # todo: check the image and recognition res when motion occurs 
    print('yolov3 res:', res_str)

    if res_str == '':
        return {}

    res_list = res_str.split('\t')
    res_dict = {}
    for key in res_list:
        location = key.split('(')[0]
        prob = key.split('(')[1].split(')')[0]
        res_dict[location] = res_dict.get(location, 0) + 1
    
    # sd = sorted(res_dict.items(), reverse=False)
    sd = sorted(res_dict.items(), key=sorter_take_count, reverse=True)

    # print(res_dict.items())

    # for k,v in sd:
    #     print('res2:', k, ' v:', v)
    #     res2 = k
    #     break

    # if res != res2:
    #     print('res:', res, ' res2:', res2)
    #     res = res2

    #print('res_activity_list:', res_activity_list)

    return sd

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
def get_location_by_activity_cnn(res_file, time_str):
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

 # get the results
    res_str = tools_ascc.read_res_from_file(res_file)

    """
    class_names=['bathroom','bedroom', 'morning_med', 'reading', 'kitchen','livingroom', 'chores', 'desk_activity', 'dining_room_activity',
                 'eve_med', 'leaving_home', 'meditate']
    """
    # use res_str[0]
    # todo: how to get the result from 10 images, for example, in the dir 08-45-46
    # todo: how to check the motion
    # todo: meditate should be treat as the bedroom activity
    # todo: re-train the model, use more imges
    # todo: check the code, how to get expected activity,  Miss activity: Expected: Sleep ,Detect: Meditate Running time: 2009-12-11 08:45:26
    # todo: check the image and recognition res when motion occurs 
    print('vision_dnn res:', res_str)

    if res_str == '':
        return '', -1

    res_list = res_str.split('\t')
    res_dict = {}

    max_location_prob = -1
    res_location = ''
    for key in res_list:
        location = key.split('(')[0]
        prob = key.split('(')[1].split(')')[0]

        if float(prob) > max_location_prob:
            res_location = location
            max_location_prob = float(prob)

        res_dict[location] = res_dict.get(key, 0) + 1

    res = res_location

    res_list = res_str.split('\t')
    res_dict = {}
    for key in res_list:
        location = key.split('(')[0]
        prob = key.split('(')[1].split(')')[0]
        res_dict[location] = res_dict.get(location, 0) + 1
    
    # sd = sorted(res_dict.items(), reverse=False)
    sd = sorted(res_dict.items(), key=sorter_take_count, reverse=True)

    # print(res_dict.items())

    for k,v in sd:
        print('res2:', k, ' v:', v)
        res2 = k
        break

    if res != res2:
        print('res:', res, ' res2:', res2)
        res = res2

    #print('res_activity_list:', res_activity_list)


    res = tools_ascc.ACTIVITY_LOCATION_MAPPING[res]

    # tools_ascc.get_activity_by_vision_dnn(time_str, action='vision')

    location, prob = res, max_location_prob
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

def get_motion_type_by_activity_cnn(res_file, time_str):

    # Mapping
    # should be act : probability
    # /home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/src/
    # ascc_room_activity_test.py


    # get the results
    res_str = tools_ascc.read_res_from_file(res_file)

    
    print('Motion Recognition res:', res_str)


    res_list = res_str.split('\t')

    motion_type_res = []
    motion_type_prob_res = []
    for key in res_list:
        motion_type = key.split('(')[0]
        prob = key.split('(')[1].split(')')[0]
        motion_type_res.append(motion_type)
        motion_type_prob_res.append(prob)


    # motion_type = res_str.split('(')[0]
    # prob = res_str.split('(')[1].split(')')[0]

    # res = motion_type



    motion_type_list, prob_list = motion_type_res, motion_type_prob_res
    # tools_ascc.get_activity_by_motion_dnn(time_str, action='vision')
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

def get_audio_type_by_activity_cnn(res_file, time_str):
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



    # get the results
    res_str = tools_ascc.read_res_from_file(res_file)

    
    print('Audio Recognition res:', res_str, ' time:', time_str)
    if res_str == '':
        return '', -1

    res_list = res_str.split('\t')

    auido_type = res_str.split('(')[0]
    prob = res_str.split('(')[1].split(')')[0]

    res = auido_type


    # Mapping
    # should be act : probability
    # /home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/src/
    # ascc_room_activity_test.py
    audio_type, prob = res, prob
    # tools_ascc.get_activity_by_audio_dnn(time_str, action='vision')

    print('get_audio_type_by_activity_cnn time_str:', time_str, ' audio_type:', audio_type, ' prob:', prob)

    return audio_type, float(prob)
    

def check_and_wait_l_o_s_m_result():


    global g_image_recognition_flag
    global g_sound_recognition_flag
    global g_motion_recognition_flag

    global g_image_recognition_file
    global g_image_recognition_time
    global g_sound_recognition_file
    global g_sound_recognition_time
    global g_motion_recognition_file
    global g_motion_recognition_time

    global g_image_object_recognition_flag
    global g_image_object_recognition_file
    global g_image_object_recognition_time


    start = timer()
    while(True):

        # if g_motion_recognition_flag:
        #     return True
            
        if g_image_recognition_flag and g_motion_recognition_flag and g_sound_recognition_flag and g_image_object_recognition_flag:
            return True
        
        end = timer()

        if (end-start) > CHECK_AND_WAIT_THRESHOLD:
            print("Get_prediction losm time out cost:", end-start)  

            break

    return False

def check_and_wait_motion_result():


    global g_image_recognition_flag
    global g_sound_recognition_flag
    global g_motion_recognition_flag

    global g_image_recognition_file
    global g_image_recognition_time
    global g_sound_recognition_file
    global g_sound_recognition_time
    global g_motion_recognition_file
    global g_motion_recognition_time

    global g_image_object_recognition_flag
    global g_image_object_recognition_file
    global g_image_object_recognition_time

    start = timer()
    while(True):
        if g_motion_recognition_flag:
            return True
        
        end = timer()
        # print("Get_prediction time cost:", end-start)  

        if (end-start) > CHECK_AND_WAIT_THRESHOLD:
            print("Get_prediction moiton time out cost:", end-start)  

            break

    return False




def real_time_test_run():


    global g_image_recognition_flag
    global g_sound_recognition_flag
    global g_motion_recognition_flag

    global g_image_recognition_file
    global g_image_recognition_time
    global g_sound_recognition_file
    global g_sound_recognition_time
    global g_motion_recognition_file
    global g_motion_recognition_time

    global g_image_object_recognition_flag
    global g_image_object_recognition_file
    global g_image_object_recognition_time

    global g_image_data_location

    global g_motion_data_location
    global g_audio_data_location

    env = real_time_env_ascc.EnvASCC(TEST_BASE_DATE + ' 00:00:00')
    # env.reset()

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
    
    new_activity_check_times = 2


    location_res = []
    audio_type_res = []
    motion_type_res = []
    object_res = []

    res_prob_audio_motion = []


    transition_motion_occur = []

    # init
    for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
        res_prob[act] = []

    object_dict = {}


    while(pre_activity == ''):
        # open camera
        print('pre_activity == empty')
        status = env.step(real_time_env_ascc.FUSION_ACTION)  # real_time_env_ascc.FUSION_ACTION?

        # env.running_time
        # test_time_str = '2009-12-11 12:58:33'
        cur_time = env.get_running_time()
        cur_time_str = cur_time.strftime(real_time_env_ascc.DATE_HOUR_TIME_FORMAT)
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


        if check_and_wait_l_o_s_m_result() == False:
            # pass
            continue


        # if not (g_image_recognition_flag and g_motion_recognition_flag and g_sound_recognition_flag and g_image_object_recognition_flag):
        #     continue


        location, location_prob = get_location_by_activity_cnn(g_image_recognition_file, g_image_recognition_time)
        bayes_model_location.set_location_prob(location_prob)

        # tools_ascc.ASCC_DATA_YOLOV3_RES_FILE
        object_dict = get_object_by_activity_yolo(g_image_object_recognition_file, g_image_object_recognition_time)
        # bayes_model_object.set_object_prob(object_prob)

        audio_type, audio_type_prob = get_audio_type_by_activity_cnn(g_sound_recognition_file, g_sound_recognition_time)
        bayes_model_audio.set_audio_type_prob(float(audio_type_prob))

        motion_type, motion_type_prob = get_motion_type_by_activity_cnn(g_motion_recognition_file, cur_time_str)
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

        # activity = cur_activity
        # time = cur_time_str
        # image_source = location
        # sound_source = audio_type
        # motion_source = motion_type
        # image_dir = g_image_recognition_file
        # TODO: the recognition model should send the image dir file as well
        # tools_sql.insert_adl_activity_data(activity, time, image_source, sound_source, motion_source)
        # print('insert int to db: activity:', activity, ' cur_time:', cur_time_str)

        
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

        g_image_recognition_flag = False 
        g_motion_recognition_flag = False
        g_sound_recognition_flag = False
        g_image_object_recognition_flag = False


        cur_g_image_recognition_time = datetime.strptime(g_image_recognition_time, FOLDER_DATE_TIME_FORMAT)
        cur_g_image_recognition_time_str = cur_g_image_recognition_time.strftime(real_time_env_ascc.DATE_HOUR_TIME_FORMAT)
        activity = cur_activity
        time = cur_g_image_recognition_time_str
        image_source = location + LOCATION_DIR_SPLIT_SYMBOL + g_image_data_location
        sound_source = audio_type
        motion_source = motion_type
        tools_sql.insert_adl_activity_data(activity, time, image_source, sound_source, motion_source)
        



        print('insert int to db: activity:', activity, ' cur_time:', cur_time_str, ' g_image_recognition_time:', cur_g_image_recognition_time_str)
            
        # TODO top3 data
        # TODO how to get the accuracy

    need_recollect_data = False
    p_check_level = 4
    start_check_interval = 0
    p_less_than_threshold_check_cnt = 0
    start_check_interval_time = None
    living_room_check_times = LIVING_ROOM_CHECK_TIMES_MAX 


    # while(not env.done):
    while(True):

        # TODO:
        # p = rank1_res_prob[-1]
        # if p of previous detection is smaller than threshodl, env.step(real_time_env_ascc.FUSION_ACTION)

        location = ''
        object = ''
        motion_type = ''
        audio_type = ''
        object_dict = {}

        living_room_check_flag = False

        if need_recollect_data:
            status = env.step(real_time_env_ascc.FUSION_ACTION)

            print("--------------------------------------------Running, FUSION_ACTION")

            # check and wait the result
            if check_and_wait_l_o_s_m_result() == False:
                continue

            # if not (g_image_recognition_flag and g_motion_recognition_flag and g_sound_recognition_flag and g_image_object_recognition_flag):
            #     continue


            location, location_prob = get_location_by_activity_cnn(g_image_recognition_file, g_image_recognition_time)
            bayes_model_location.set_location_prob(location_prob)

            # tools_ascc.ASCC_DATA_YOLOV3_RES_FILE
            object_dict = get_object_by_activity_yolo(g_image_object_recognition_file, g_image_object_recognition_time)
            # bayes_model_object.set_object_prob(object_prob)

            audio_type, audio_type_prob = get_audio_type_by_activity_cnn(g_sound_recognition_file, g_sound_recognition_time)
            bayes_model_audio.set_audio_type_prob(float(audio_type_prob))

            motion_type, motion_type_prob = get_motion_type_by_activity_cnn(g_motion_recognition_file, g_motion_recognition_time)
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



            # # todo wait the event then read the recognition result  
            # location, location_prob = get_location_by_activity_cnn(cur_time_str)
            # bayes_model_location.set_location_prob(location_prob)

            # object_dict = get_object_by_activity_yolo(cur_time_str)
            # # bayes_model_object.set_object_prob(object_prob)
            # object = object_dict

            # audio_type, audio_type_prob = get_audio_type_by_activity_cnn(cur_time_str)
            # bayes_model_audio.set_audio_type_prob(audio_type_prob)

            # motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
            # bayes_model_motion.set_motion_type_prob(motion_type_prob)
            # # bayes_model_motion.set_motion_type(motion_type)


            # location_res.append([location, location_prob])
            # audio_type_res.append([audio_type, audio_type_prob])
            # # motion_type_res.append([motion_type, motion_type_prob])

            # if location == '':
            #     print('Location  empty:', cur_time)
            #     cur_time = env.get_running_time()
            #     cur_time_str = cur_time.strftime(real_time_env_ascc.DATE_HOUR_TIME_FORMAT)
            #     continue

        else:
            # INTERVAL_FOR_COLLECTING_DATA
            status = env.step(real_time_env_ascc.MOTION_ACTION)  
            print("--------------------------------------------Running, MOTION_ACTION")

            # check and wait the result
            if check_and_wait_motion_result() == False:
                continue

            # if not (g_motion_recognition_flag):
            #     continue

            motion_type, motion_type_prob = get_motion_type_by_activity_cnn(g_motion_recognition_file, g_motion_recognition_time)
            bayes_model_motion.set_motion_type_prob(motion_type_prob)
            # bayes_model_motion.set_motion_type(motion_type)

            # # detect transition: the end of walk activity
            # motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
            # bayes_model_motion.set_motion_type_prob(motion_type_prob)

            location_res.append(['', ''])
            audio_type_res.append(['', ''])
            # motion_type_res.append([motion_type, motion_type_prob])
        
        g_image_recognition_flag = False 
        g_motion_recognition_flag = False
        g_sound_recognition_flag = False
        g_image_object_recognition_flag = False





        heap_prob = []
        heap_prob_audio_motion = []
        # if transition_motion:
        #     cur_time = env.get_running_time()
        #     cur_time_str = cur_time.strftime(real_time_env_ascc.DATE_HOUR_TIME_FORMAT)
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
        cur_time_str = cur_time.strftime(real_time_env_ascc.DATE_HOUR_TIME_FORMAT)
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
        print('object:', object_dict)
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
                # p4 = bayes_model_object.get_prob(pre_act_list, act, object, activity_duration)

                p4 = 1
                # p3 = 1
                
                p_audio_motion = p2 * p3 * hmm_prob

                # todo, in the living room, we can collect audio data more times (3 -r times), to confirm the activity
                # or, we can get object activity
                # object == constants.OBJECT_BOOK
                #if audio_type == constants.AUDIO_TYPE_ENV:
            


                if location == constants.LOCATION_LIVINGROOM:

                    # todo: if read, watch_tv, desk_activity detected, recollect data after 1 min to double check the activity
                    # or set a delay to recollect the data, this is not good because some blurred image maybe miss recognition to living room

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

                    
                
                p = p1*p2*p3*p4 * hmm_prob

                
                print("need_recollect_data step act:", act)
                print('hmm_prob:', hmm_prob)
                print('p1:', p1)
                print('p2:', p2)
                print('p3:', p3)
                print('p4:', p4)
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

            # if p_activity_end < -0.3:    
            #     if start_check_interval_time == None:
            #         start_check_interval_time = cur_time

            #     start_check_interval = (cur_time - start_check_interval_time).seconds 
            #     print('start_check_interval:', start_check_interval, ' start_check_interval_time:', start_check_interval_time)

            #     if (int(start_check_interval) / UNCERTAIN_CHECK_INTERVAL) >= 1:
            #         need_recollect_data = True
            #         print('reset start_check_interval:', start_check_interval, ' start_check_interval_time:', start_check_interval_time)
            #         start_check_interval = 0
            #         start_check_interval_time = cur_time

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
            time = g_image_recognition_time
            image_source = location + LOCATION_DIR_SPLIT_SYMBOL + g_image_data_location

            sound_source = audio_type
            motion_source = motion_type
            object_source = ''

            if location == constants.LOCATION_LIVINGROOM:
                for object, prob in object_dict:
                    object_source = object_source + '_' + object

            tools_sql.insert_adl_activity_data(activity, time, image_source, sound_source, motion_source, object_source)
            print('insert int to db: activity:', activity, ' cur_time:', cur_time_str)


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

        print('last motion type:', motion_type_res[-1], ' cur motion_type:', motion_type)
        if (motion_type_res[-1][0] == motion_adl_bayes_model.MOTION_TYPE_WALKING) and (motion_type != motion_adl_bayes_model.MOTION_TYPE_WALKING):
            print('Transition occur:', 'last motion type:', motion_type_res[-1], ' cur motion_type:', motion_type)
            transition_motion = TRUE
            need_recollect_data = True
            transition_motion_occur.append(cur_time_str)

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

        global g_stop

        if g_stop:
            break



    # while not env.done


    print("Final res ===================================================")
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

    print("Sensors Sensor Time cost:", env.sensor_time_cost)
    print("Sensors Energy total  cost:", env.sensor_energy_cost)
    print("Total total_check_times:", env.total_check_times)
    print("Total motion_check_times:", env.motion_check_times)
    print("Total fusion_check_times:", env.fusion_check_times)
    end_time_of_wmu = datetime.strptime(env.get_current_running_time().strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
    print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)

    # # motion probabilities during activities
    # print('p_sitting_prob:', len(p_sitting_prob))
    # print(p_sitting_prob)

    # print('p_standing_prob:', len(p_standing_prob))
    # print(p_standing_prob)

    # print('p_walking_prob:', len(p_walking_prob))
    # print(p_walking_prob)

    # todo: probability of each activities obtained from the p_duration, for example, cur_activity is 'Read', P_duration(Read) = 0.8, then p(rank2) + p(rank3) = 1-p(Read)=1- 0.8  = 0.2



    print("===================================================")

# if env.done:
#     print("Activity_none_times:", env.activity_none_times)
#     print("Expected_activity_none_times:", env.expected_activity_none_times)
#     print("Hit times:", env.done_hit_event_times)
#     print("Miss times:", env.done_missed_event_times)
#     print("Random Miss times:", env.done_random_miss_event_times)
#     print("Middle times:", env.done_middle_event_times)
#     print("Penalty times:", env.done_penalty_times)
#     print("Uncertain times:", env.done_uncertain_times)
#     print("Total times:", env.done_totol_check_times)
#     print("Residual power:", env.done_residual_power)
#     print("Beginning event times:", env.done_beginning_event_times)
#     print("Endding event times:", env.done_end_event_times)
#     print("Middle event times:", env.done_middle_event_times)
#     print("Day End Running time:", env.done_running_time)
#     print("Reward:", env.done_reward)
#     print("Done status:", env.done)
#     print("Sensors Energy cost:", env.done_energy_cost)
#     print("Sensors Time cost:", env.done_time_cost)
#     print("Sensors Energy total  cost:", env.energy_cost)
#     print("Sensors Time total cost:", env.time_cost)
#     print("Total Time cost:", env.total_check_times)
#     print("Motion_triggered_times:", env.motion_triggered_times)
#     print("Hit_activity_check_times", env.hit_activity_check_times)
#     end_time_of_wmu = datetime.strptime(env.done_running_time.strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
#     print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)

# end_time_of_wmu = datetime.strptime(env.done_running_time.strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
# print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)


# print("Display information:")
# # env.display_action_counter()
# # env.display_info()
# print("===================================================")

# print("Activity_none_times \t Expected_activity_none_times \t Hit times \t Miss times \
#     \t Random Miss times \t Penalty times \t Uncertain times \t Total times \
#     \t Residual power \t Beginning event times \t Endding event times \t Middle event times \
#     \t Day End Running time \t Done status \t Duration of Day \t DoneReward \t Reward  \
#     \t Sensors energy cost \t Sensors time cost \t total time cost \t Motion_triggered_times \t Hit_activity_check_times \t motion_activity_cnt \t")

# print(env.activity_none_times, '\t', env.expected_activity_none_times, '\t',env.done_hit_event_times, '\t', env.done_missed_event_times, \
#     '\t', env.done_random_miss_event_times, '\t', env.done_penalty_times, '\t', env.done_uncertain_times, '\t', env.done_totol_check_times, \
#     '\t', env.done_residual_power, '\t', env.done_beginning_event_times, '\t', env.done_end_event_times, '\t', env.done_middle_event_times, \
#     '\t', env.done_running_time, '\t', env.done, '\t', (end_time_of_wmu - env.day_begin).seconds/3600.0, '\t', 0, '\t', 0, \
#     '\t', env.done_energy_cost, '\t', env.done_time_cost, "\t", env.done_total_time_cost, "\t", env.motion_triggered_times, '\t', env.hit_activity_check_times, '\t',env.motion_activity_cnt)

# print("===================================================")


import adl_env_client_lib
import asyncio
import signal
import socketio
import functools
import time


# Update the IP Address according the target server
IP_ADDRESS = 'http://127.0.0.1:5000'
# Update your group ID
GROUP_ID = 1

INTERVAL = 10

shutdown = False


DATA_FILE_RECEIVED_FROM_WMU_EVENT_NAME = 'DATA_FILE_RECEIVED_FROM_WMU'
DATA_RECOGNITION_FROM_WMU_EVENT_NAME = 'DATA_RECOGNITION_FROM_WMU'

DATA_RECOGNITION_FINAL_TO_ADL_EVENT_NAME = 'DATA_RECOGNITION_TO_ADL'


DATA_TYPE = 'type'
DATA_CURRENT = 'current_time'
DATA_FILE = 'file'
DATA_TYPE_IMAGE = 'image'
DATA_TYPE_SOUND = 'audio'
DATA_TYPE_MOTION = 'motion'
DATA_TYPE_IMAGE_YOLO = 'yolo'
DATA_LOCATION = 'data_location'


STOP_ADL_SERVER = 'stop_adl_server'

# For getting the score
sio = socketio.AsyncClient()

@sio.event
async def connect():
    print('connection established')

@sio.on(STOP_ADL_SERVER)
async def on_message(data):
    print('Get STOP_ADL_SERVER notice:', data)
    global g_stop
    g_stop = True

@sio.on(DATA_RECOGNITION_FINAL_TO_ADL_EVENT_NAME)
async def on_message(data):
    try:
        if data['type'] == DATA_TYPE_IMAGE:
            print('Get image recognition:', data)
            global g_image_recognition_flag
            global g_image_recognition_file
            global g_image_recognition_time

            global g_image_data_location

            g_image_recognition_flag = True
            g_image_recognition_file = data[DATA_FILE]
        
            cur_time = data[DATA_CURRENT]
            file = data[DATA_FILE]
            g_image_recognition_time = cur_time


            g_image_data_location = data[DATA_LOCATION]
            
            print('cur_time:', cur_time, 'file:', g_image_data_location)

        elif data['type'] == DATA_TYPE_IMAGE_YOLO:
            print('Get image yolo recognition:', data)
            global g_image_object_recognition_flag
            global g_image_object_recognition_file
            global g_image_object_recognition_time

            g_image_object_recognition_flag = True
            g_image_object_recognition_file = data[DATA_FILE]
            g_image_object_recognition_time = cur_time
        
            cur_time = data[DATA_CURRENT]
            file = data[DATA_FILE]
            
            print('cur_time:', cur_time, 'file:', file)

        elif data['type'] == DATA_TYPE_MOTION:
            print('Get motion recognition:', data)
            global g_motion_recognition_flag
            global g_motion_recognition_file
            global g_motion_recognition_time
            global g_motion_data_location
            global g_audio_data_location


            g_motion_recognition_flag = True
            g_motion_recognition_file = data[DATA_FILE]
            g_motion_recognition_time = cur_time
        
            cur_time = data[DATA_CURRENT]
            file = data[DATA_FILE]
            
            g_motion_data_location = data[DATA_LOCATION]

            print('cur_time:', cur_time, 'file:', g_motion_data_location)

        elif data['type'] == DATA_TYPE_SOUND:
            print('Get sound recognition:', data)
            global g_sound_recognition_flag
            global g_sound_recognition_file
            global g_sound_recognition_time
            global g_audio_data_location

            g_sound_recognition_flag = True
            g_sound_recognition_file = data[DATA_FILE]
            g_sound_recognition_time = cur_time
        
            cur_time = data[DATA_CURRENT]
            file = data[DATA_FILE]

            g_audio_data_location = data[DATA_LOCATION]

            
            print('cur_time:', cur_time, 'file:', g_audio_data_location)



    except:
        pass
    print('Got final recognition data:', data)


@sio.event
async def disconnect():
    print('disconnected from server')

def stop(signame, loop):
    global shutdown
    shutdown = True

    tasks = asyncio.all_tasks()
    for _task in tasks:
        _task.cancel()

async def run():
    cnt = 0
    global shutdown
    while not shutdown:
        print('.', end='', flush=True)

        try:
            await asyncio.sleep(INTERVAL)
            cnt = cnt + INTERVAL
            print('run: ', cnt)
            # event_name = DATA_RECOGNITION_FROM_WMU_EVENT_NAME
            # broadcasted_data = {'type': DATA_TYPE_IMAGE, 'file': 'image0'}
            # await sio.emit(event_name, broadcasted_data)
        except asyncio.CancelledError as e:
            pass
            #print('run', 'CancelledError', flush=True)

    await sio.disconnect()

async def main():
    await sio.connect(IP_ADDRESS)

    loop = asyncio.get_running_loop()

    for signame in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signame),
            functools.partial(stop, signame, loop))

    task = asyncio.create_task(run())
    try:
        await asyncio.gather(task)
    except asyncio.CancelledError as e:
        pass
        #print('main', 'cancelledError')

    print('main-END')


if __name__ == '__main__':

    real_time_server = threading.Thread(target=real_time_test_run)
    # # web_server = threading.Thread(target=web_server_run)

    real_time_server.start()
    # # web_server.start()

    # # server.join()

    asyncio.run(main())
    