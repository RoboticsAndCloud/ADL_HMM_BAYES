"""
Brief: Bayes model for activity probability, including vision, motion, audio-based model.
Author: Frank
Date: 07/10/2022
"""

import motion_env_ascc
import motion_adl_bayes_model
import random

import tools_ascc
import hmm


MILAN_BASE_DATE = '2009-10-16'

MILAN_BASE_DATE_HOUR = '2009-10-16 06:00:00'

TEST_BASE_DATE = '2009-12-11'


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

    sd = sorted(act_dict.items(), reverse=True)
    res = sd[0][0]

    random_t = random.random()
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

    sd = sorted(act_dict.items(), reverse=True)
    res = sd[0][0]

    random_t = random.random()
    print('random_t:', random_t)
    if random_t > sd[0][1] and (len(sd) > 1):
        index = random.randint(1, len(sd)-1)
        res = sd[index][0]

    return res

def get_motion_type_by_activity(activity):
    # motion type: sitting, standing, walking, random by the probs

        # Mapping
    act_dict = motion_adl_bayes_model.P2_Motion_type_Under_Act[activity]

    sd = sorted(act_dict.items(), reverse=True)
    res = sd[0][0]

    random_t = random.random()
    print('random_t:', random_t)
    if random_t > sd[0][1] and (len(sd) > 1):
        index = random.randint(1, len(sd)-1)
        res = sd[index][0]

    return res

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

    sd = sorted(act_dict.items(), reverse=True)
    res = sd[0][0]

    random_t = random.random()
    print('random_t:', random_t)
    if random_t > sd[0][1] and (len(sd) > 1):
        index = random.randint(1, len(sd)-1)
        res = sd[index][0]
    
    return res


env = motion_env_ascc.EnvASCC(TEST_BASE_DATE + ' 00:00:00')
env.reset()

hmm_model = get_hmm_model()

bayes_model_location = motion_adl_bayes_model.Bayes_Model_Vision_Location(hmm_model=hmm_model, simulation=True)
bayes_model_motion = motion_adl_bayes_model.Bayes_Model_Motion(hmm_model=hmm_model, simulation=True)
bayes_model_audio = motion_adl_bayes_model.Bayes_Model_Audio(hmm_model=hmm_model, simulation=True)
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

pre_act_list = []

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

    cur_activity, cur_beginning_activity, cur_end_activity = \
        bayes_model_location.get_activity_from_dataset_by_time(cur_time_str)

    print('cur_time:', cur_time, ' cur_activity:', cur_activity)
    # exit(0)

    """
    2009-12-11 08:42:03.000082	M021	ON	Sleep end
    2009-12-11 08:42:04.000066	M028	ON
    2009-12-11 08:42:06.000089	M020	ON
    """
    if cur_activity == None or cur_activity == '':
        continue


    # detect activity, cur_activity, pre_activity
    # Bayes model
    location = get_location_by_activity(cur_activity)
    object = get_object_by_activity(cur_activity)
    audio_type = get_audio_type_by_activity(cur_activity)
    motion_type = get_motion_type_by_activity(cur_activity)
    
    heap_prob = []

    for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
        p1 = bayes_model_location.get_prob(pre_act_list, act, location, 0)
        
        #  bayes_model_location.prob_of_location_under_act(location, act) \
            #  * motion_adl_bayes_model.HMM_START_MATRIX[act] /(bayes_model_location.prob_of_location_under_all_acts(location)) * bayes_model_location.prob_of_location_using_vision(location, act)
        
        p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, 0)
        p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, 0)
        p4 = bayes_model_object.get_prob(pre_act_list, act, object, 0)

        p = p1*p2*p3*p4
             
        res_prob[act].append(p)
        heap_prob.append((act, p, cur_time_str))
        
    print('heap_prob:', heap_prob)
    top3_prob = sorted(heap_prob, key=sorter_take_count,reverse=True)[:3]
    print('top3_prob:', top3_prob)


    rank1_res_prob.append(top3_prob[0])
    rank2_res_prob.append(top3_prob[1])
    rank3_res_prob.append(top3_prob[2])

    pre_activity = top3_prob[0][0]
    cur_activity = top3_prob[0][0]
    activity_begin_time = cur_time

    pre_act_list.append(pre_activity)
    # TODO top3 data
    # TODO how to get the accuracy

need_recollect_data = False
while(not env.done):

    # TODO:
    # p = rank1_res_prob[-1]
    # if p of previous detection is smaller than threshodl, env.step(motion_env_ascc.FUSION_ACTION)

    if need_recollect_data:
        audio_data, vision_data, motion_data, transition_motion = env.step(motion_env_ascc.AUDIO_ACTION)  
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
            
        if cur_activity == None or cur_activity == '':
            continue

        location = get_location_by_activity(cur_activity)
        object = get_object_by_activity(cur_activity)
        audio_type = get_audio_type_by_activity(cur_activity)
        motion_type = get_motion_type_by_activity(cur_activity)
        
        activity_duration = (cur_time - activity_begin_time).seconds / 60 # in minutes


        for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
            p1 = bayes_model_location.get_prob(pre_act_list, act, location, activity_duration)
            p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, activity_duration)
            p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, activity_duration)
            p4 = bayes_model_object.get_prob(pre_act_list, act, object, activity_duration)
            p = p1*p2*p3*p4

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

        if cur_activity == None or cur_activity == '':
            continue
        
        activity_duration = (cur_time - activity_begin_time).seconds / 60 # in minutes
        # prob_of_activity_by_duration = get_end_of_activity_prob_by_duration(activity_duration, cur_activity)

        location = get_location_by_activity(cur_activity)
        motion_type = get_motion_type_by_activity(cur_activity)
        audio_type = get_audio_type_by_activity(cur_activity)

        for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():

            p2 = bayes_model_motion.get_prob(pre_act_list, act, motion_type, activity_duration)

            if need_recollect_data:
                p3 = bayes_model_audio.get_prob(pre_act_list, act, audio_type, activity_duration)
                # todo: audio_type: quiet, ignore this or not? reading? cooking?
                p = p2 * p3

            p = p2

            res_prob[act].append(p) 
            heap_prob.append((act, p, cur_time_str))
        print('heap_prob:', heap_prob)
        top3_prob = sorted(heap_prob, key=sorter_take_count,reverse=True)[:3]
        print('top3_prob:', top3_prob)
        # TODO: normalization for the top3 prob
        # if top3_prob[0] < threshold:
        #     need_recollect_data = True

        rank1_res_prob.append(top3_prob[0])
        rank2_res_prob.append(top3_prob[1])
        rank3_res_prob.append(top3_prob[2])

        # todo, if rank1 - rank2 < 0.001, p= p+ p*p_audio, to get a more accurate res
        #        
        cur_activity = top3_prob[0][0]

        if pre_activity != cur_activity:
            pre_activity = cur_activity
            pre_act_list.append(pre_activity)
            activity_begin_time = cur_time
            need_recollect_data = False


# print out results
print('rank1:')
print(rank1_res_prob)

print('rank2:')
print(rank2_res_prob)

print('rank3:')
print(rank3_res_prob)

print('res_prob:')
print(res_prob)



