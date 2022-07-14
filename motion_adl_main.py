"""
Brief: Bayes model for activity probability, including vision, motion, audio-based model.
Author: Frank
Date: 07/10/2022
"""

import motion_env_ascc
import motion_adl_bayes_model
import tools_ascc


MILAN_BASE_DATE = '2009-10-16'

MILAN_BASE_DATE_HOUR = '2009-10-16 06:00:00'

TEST_BASE_DATE = '2009-12-11'


"""
Given the duration, return the probability that the activity may be finished
For Example: Read, we got 20mins(5 times), 30mins(10), 40mins(25), 60mins(2),  for duration of 20mins, the probability would be 5/(5+10+25+2) = 11.90% 
"""
act_duration_cnt_dict = tools_ascc.get_activity_duration_cnt_set()
def get_end_of_activity_prob_by_duration(activity_duration, activity):
    d_lis = act_duration_cnt_dict[activity]
    total_cnt = len(d_lis)
    cnt = 0
    for d in d_lis:
        if activity_duration >= d:
            cnt += 1
    
    prob = 1 - cnt * 1.0 /total_cnt

    if prob < 0.01:
        prob = 0.01

    return prob


def get_location_by_activity(activity):
    return activity

def get_motion_type_by_activity(activity):
    # motion type: sitting, standing, walking, random by the probs

    return activity

def get_audio_type_by_activity(activity):
    # audio type:
    # door_open_closed
    # drinking, eating
    # flush_toilet
    # keyboard
    # microwave
    # pouring_water_into_glass
    # quiet
    # toothbrushing
    # tv_news
    # vacuum
    # washing_hand
    
    return activity



env = motion_env_ascc.EnvASCC(TEST_BASE_DATE + '00:00:00')
env.reset()

bayes_model_vision = motion_adl_bayes_model.Bayes_Model_Vision()
bayes_model_motion = motion_adl_bayes_model.Bayes_Model_Motion()

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

for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
    res_prob[act] = []

if pre_activity == '':
    # open camera

    audio_data, vision_data, motion_data, transition_motion = env.step(motion_env_ascc.VISION_ACTION)  # motion_env_ascc.FUSION_ACTION?

    # env.running_time
    # test_time_str = '2009-12-11 12:58:33'
    cur_time = env.get_running_time()
    cur_activity, cur_beginning_activity, cur_end_activity = \
        bayes_model_vision.get_activity_from_dataset_by_time(cur_time)

    # detect activity, cur_activity, pre_activity
    # Bayes model
    location = cur_activity
    # prob = bayes_model_vision.get_prob(pre_activity, cur_activity, location)
    
    heap_prob = []

    for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
        p =  bayes_model_vision.prob_of_location_under_act(location, act) \
             * motion_adl_bayes_model.HMM_START_MATRIX[act] /(bayes_model_vision.prob_of_location_under_all_acts(location)) * bayes_model_vision.prob_of_location_using_vision(location, act)
             
        res_prob[act].append(p)
        heap_prob.append((act, p, cur_time))


    top3_prob = sorted(heap_prob, reverse=True)[:3]
    rank1_res_prob.append(top3_prob[0])
    rank2_res_prob.append(top3_prob[1])
    rank3_res_prob.append(top3_prob[2])
    pre_activity = top3_prob[0][1]
    cur_activity = top3_prob[0][1]
    activity_begin_time = cur_time
    # TODO top3 data
    # TODO how to get the accuracy


while(not env.done):

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
        print('Env Running:', cur_time) 
        cur_activity, cur_beginning_activity, cur_end_activity = \
            bayes_model_vision.get_activity_from_dataset_by_time(cur_time)

        location = cur_activity
        for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
            p1 =  bayes_model_vision.get_prob(pre_activity, act, location)

            # p2 = bayes_model_audio.get_prob(pre_activity, act, location)
            # p3 = bayes_model_motion.get_prob(pre_activity, act, location)

            # calculate the duration of activity to get the probability


            res_prob[act].append(p1) 
            heap_prob.append((act, p, cur_time))

            # p = p1 + p1*p2 + p1*p3
            # todo print out top 3

            # pre_activity should be the acitivty with high probability
            pre_activity = cur_activity
            activity_begin_time = cur_time

    else:
        # motion data  or audio data
        # new activity
        # Bayes model prob    

        cur_time = env.get_running_time()
        print('Env Running:', cur_time) 
        cur_activity, cur_beginning_activity, cur_end_activity = \
            bayes_model_vision.get_activity_from_dataset_by_time(cur_time)

        
        activity_duration = (cur_time - activity_begin_time).seconds() / 60 # in minutes
        prob_of_activity_by_duration = get_end_of_activity_prob_by_duration(activity_duration, cur_activity)

        location = get_location_by_activity(cur_activity)
        motion_type = get_motion_type_by_activity(cur_activity)
        audio_type = get_audio_type_by_activity(cur_activity)
        for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():

            p =  bayes_model_motion.get_prob(pre_activity, act, motion_type) 
            p = p * prob_of_activity_by_duration
            # p3 = bayes_model_audio.get_prob(pre_activity, cur_activity, location)


            res_prob[act].append(p) 
            heap_prob.append((act, p, cur_time))

            # p = p2 + p2*p3
            # todo print out top 3

            # pre_activity should be the acitivty with high probability
            pre_activity = cur_activity
    

        top3_prob = sorted(heap_prob, reverse=True)[:3]
        rank1_res_prob.append(top3_prob[0])
        rank2_res_prob.append(top3_prob[1])
        rank3_res_prob.append(top3_prob[2])

        # todo, if rank1 - rank2 < 0.001, p= p+ p*p_audio, to get a more accurate res
        #

        pre_activity = top3_prob[0][1]
        cur_activity = top3_prob[0][1]





