"""
Brief: Bayes model for activity probability, including vision, motion, audio-based model.
Author: Frank
Date: 07/10/2022
"""

from datetime import datetime
import random
#from tkinter.messagebox import NO

import hmm
import motion_env_ascc
import motion_adl_bayes_model
import tools_ascc





MILAN_BASE_DATE = '2009-10-16'

MILAN_BASE_DATE_HOUR = '2009-10-16 06:00:00'

TEST_BASE_DATE = '2009-12-11'
DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
HOUR_TIME_FORMAT = "%H:%M:%S"
DAY_FORMAT_STR = '%Y-%m-%d'

UNCERTAIN_CHECK_INTERVAL = 60 # Seconds


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

    return location, prob

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
    motion_type, prob = tools_ascc.get_activity_by_motion_dnn(time_str, action='vision')

    print('get_motion_type_by_activity_cnn time_str:', time_str, ' motion_type:', motion_type, ' prob:', prob)

    return motion_type, prob

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

    return audio_type, prob
    


env = motion_env_ascc.EnvASCC(TEST_BASE_DATE + ' 00:00:00')
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

location_res = []
audio_type_res = []
motion_type_res = []

def get_pre_act_list():

    return []

# init
for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
    res_prob[act] = []

while(pre_activity == ''):
    # open camera

    audio_data, vision_data, motion_data, transition_motion = env.step(motion_env_ascc.VISION_ACTION)  # motion_env_ascc.FUSION_ACTION?

    # env.running_time
    # test_time_str = '2009-12-11 12:58:33'
    cur_time = env.get_running_time()
    cur_time_str = cur_time.strftime(motion_env_ascc.DATE_HOUR_TIME_FORMAT)
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

    location, location_prob = get_location_by_activity_cnn(cur_time_str)
    bayes_model_location.set_location_prob(location_prob)

    # object, object_prob = get_object_by_activity(cur_time_str)
    # bayes_model_object.set_location_prob(object_prob)

    audio_type, audio_type_prob = get_audio_type_by_activity_cnn(cur_time_str)
    bayes_model_audio.set_audio_type_prob(audio_type_prob)

    motion_type, motion_type_prob = get_motion_type_by_activity_cnn(cur_time_str)
    bayes_model_motion.set_motion_type_prob(motion_type_prob)
    # bayes_model_motion.set_motion_type(motion_type)

    location_res.append([location, location_prob])
    audio_type_res.append([audio_type, audio_type_prob])
    motion_type_res.append([motion_type, motion_type_prob])

    exit(0)



    
    heap_prob = []

    p2_res_dict = {}

    for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
        hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_list, act, activity_duration)

        p1 = bayes_model_location.get_prob(pre_act_list, act, location, 0)
        
        #  bayes_model_location.prob_of_location_under_act(location, act) \
            #  * motion_adl_bayes_model.HMM_START_MATRIX[act] /(bayes_model_location.prob_of_location_under_all_acts(location)) * bayes_model_location.prob_of_location_using_vision(location, act)
        
        p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, 0)
        p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, 0)
        p4 = bayes_model_object.get_prob(pre_act_list, act, object, 0)

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
    # TODO top3 data
    # TODO how to get the accuracy

need_recollect_data = False
p_check_level = 4
start_check_interval = 0
while(not env.done):

    # TODO:
    # p = rank1_res_prob[-1]
    # if p of previous detection is smaller than threshodl, env.step(motion_env_ascc.FUSION_ACTION)

    if need_recollect_data:
        audio_data, vision_data, motion_data, transition_motion = env.step(motion_env_ascc.FUSION_ACTION)  
    else:
        # INTERVAL_FOR_COLLECTING_DATA
        audio_data, vision_data, motion_data, transition_motion = env.step(motion_env_ascc.MOTION_ACTION)  

    # detect motion
    #if transition_motion:
        # open all sensors
        # new activity
        # Bayes model prob

    heap_prob = []
    if transition_motion:
        cur_time = env.get_running_time()
        cur_time_str = cur_time.strftime(motion_env_ascc.DATE_HOUR_TIME_FORMAT)
        print('Env Running:', cur_time) 
        
        bayes_model_location.set_time(cur_time_str)
        bayes_model_motion.set_time(cur_time_str)
        bayes_model_audio.set_time(cur_time_str)
        bayes_model_object.set_time(cur_time_str)

        cur_activity, cur_beginning_activity, cur_end_activity = \
            bayes_model_location.get_activity_from_dataset_by_time(cur_time_str)
            
        if cur_activity == None or cur_activity == '' or cur_activity == 'Sleep':
            continue

        location = get_location_by_activity(cur_activity)
        object = get_object_by_activity(cur_activity)
        audio_type = get_audio_type_by_activity(cur_activity)
        motion_type = get_motion_type_by_activity(cur_activity)

        print('location:', location)
        print('object:', object)
        print('audio_type:', audio_type)
        print('motion_type:', motion_type)

        # if pre_activity != cur_activity:
        #     activity_begin_time = cur_time
        
        activity_duration = (cur_time - activity_begin_time).seconds / 60 # in minutes


        for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
            print("transition step act:", act)
            hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_list, act, activity_duration)

            p1 = bayes_model_location.get_prob(pre_act_list, act, location, activity_duration)
            p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, activity_duration)
            p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, activity_duration)
            p4 = bayes_model_object.get_prob(pre_act_list, act, object, activity_duration)

            p = p1*p2*p3*p4 * hmm_prob
            
            print("transition step act:", act)
            print('p1:', p1)
            print('p2:', p2)
            print('p3:', p3)
            print('p4:', p4)
            print('p:', p)
            print("======================================================")

            res_prob[act].append(p) 
            heap_prob.append((act, p, cur_time_str))

    else:
        # motion data  or audio data
        # new activity
        # Bayes model prob    

        cur_time = env.get_running_time()
        cur_time_str = cur_time.strftime(motion_env_ascc.DATE_HOUR_TIME_FORMAT)
        print('Env Running:', cur_time) 

        

        bayes_model_location.set_time(cur_time_str)
        bayes_model_motion.set_time(cur_time_str)
        bayes_model_audio.set_time(cur_time_str)
        bayes_model_object.set_time(cur_time_str)

        cur_activity, cur_beginning_activity, cur_end_activity = \
            bayes_model_location.get_activity_from_dataset_by_time(cur_time_str)

        if cur_activity == None or cur_activity == '' or cur_activity == 'Sleep':
            continue
        
        activity_duration = (cur_time - activity_begin_time).seconds / 60 # in minutes
        # prob_of_activity_by_duration = get_end_of_activity_prob_by_duration(activity_duration, cur_activity)

        location = get_location_by_activity(cur_activity)
        object = get_object_by_activity(cur_activity)
        motion_type = get_motion_type_by_activity(cur_activity)
        audio_type = get_audio_type_by_activity(cur_activity)

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
            hmm_prob = bayes_model_location.prob_prior_act_by_prelist(pre_act_list, act, activity_duration)

            print('hmm_prob:', hmm_prob)

            p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, activity_duration)
            p = p2 * hmm_prob

            if need_recollect_data:
                p1 = bayes_model_location.get_prob(pre_act_list, act, location, activity_duration)
                p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, activity_duration)
                p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, activity_duration)
                p4 = bayes_model_object.get_prob(pre_act_list, act, object, activity_duration)

                p = p1*p2*p3*p4 * hmm_prob
                
                print("need_recollect_data step act:", act)
                print('p1:', p1)
                print('p2:', p2)
                print('p3:', p3)
                print('p4:', p4)

                
            print("motion step act:", act)
            print('p:', p)
            print("======================================================")

            res_prob[act].append(p) 
            heap_prob.append((act, p, cur_time_str))

        need_recollect_data = False

        top3_prob = sorted(heap_prob, key=sorter_take_count,reverse=True)[:3]
        activity_detected = top3_prob[0][0]

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

        p_activity_end = motion_adl_bayes_model.get_end_of_activity_prob_by_duration(activity_duration, activity_detected)

        p_duration_lis.append(p_activity_end)

        if activity_detected == pre_activity:
            print('p_activity_end:', p_activity_end)
            # todo if p_activity_end < 0.2, audio,vision+motion
            if (p_activity_end < 0.4) and (p_check_level == 4):
                start_check_interval_time = cur_time
                need_recollect_data = True
                p_check_level = p_check_level -1
            if (p_activity_end < 0.3) and (p_check_level == 3):
                start_check_interval_time = cur_time
                need_recollect_data = True
                p_check_level = p_check_level -1
            if (p_activity_end < 0.2) and (p_check_level == 2):
                start_check_interval_time = cur_time
                p_check_level = p_check_level -1

            if p_activity_end < 0.2:    
                start_check_interval = (cur_time - start_check_interval_time).seconds 
                print('start_check_interval:', start_check_interval, ' start_check_interval_time:', start_check_interval_time)

                if (int(start_check_interval) / UNCERTAIN_CHECK_INTERVAL) >= 1:
                    need_recollect_data = True
                    print('reset start_check_interval:', start_check_interval, ' start_check_interval_time:', start_check_interval_time)
                    start_check_interval = 0
                    start_check_interval_time = cur_time

            #     need_recollect_data = True
            #     p_check_level = p_check_level -1
            # if (p_activity_end < 0.1) and (p_check_level == 1):
            #     need_recollect_data = True
            #     p_check_level = p_check_level -1
            # if (p_activity_end < 0.05) and (p_check_level == 0):
            #     need_recollect_data = True
            #     p_check_level = p_check_level -1
            # if (p_activity_end < 0.01):
            #     need_recollect_data = True
            #     p_check_level = p_check_level -1
            print("need_recollect_data p_check_level:", need_recollect_data, ' ', p_check_level)


    print('pre_act_list:', pre_act_list)
    print('heap_prob:', heap_prob)
    top3_prob = sorted(heap_prob, key=sorter_take_count,reverse=True)[:3]
    print('top3_prob:', top3_prob)
    # TODO: normalization for the top3 prob
    # if top3_prob[0] < threshold:
    #     need_recollect_data = True

    rank1_res_prob.append(top3_prob[0])
    rank2_res_prob.append(top3_prob[1])
    rank3_res_prob.append(top3_prob[2])

    rank1_res_prob_norm.append(p_activity_end)
    p_rank2 = (1-p_activity_end) * (rank2_res_prob[-1][1] + 1e-200)/(rank2_res_prob[-1][1]+ 1e-200+rank3_res_prob[-1][1]+ 1e-200)
    rank2_res_prob_norm.append(p_rank2)
    p_rank3 = (1-p_activity_end) * (rank3_res_prob[-1][1] + 1e-200)/(rank2_res_prob[-1][1]+ 1e-200+rank3_res_prob[-1][1]+ 1e-200)
    rank3_res_prob_norm.append(p_rank3)

    # todo, if rank1 - rank2 < 0.001, p= p+ p*p_audio, to get a more accurate res
    #        
    cur_activity = top3_prob[0][0]

    if pre_activity != cur_activity:
        pre_activity = cur_activity
        pre_act_list.append(pre_activity)
        activity_begin_time = cur_time
        need_recollect_data = False
        p_check_level = 4
        start_check_interval_time = None
        start_check_interval = 0

    
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
