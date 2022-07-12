"""
Brief: Bayes model for activity probability, including vision, motion, audio-based model.
Author: Frank
Date: 07/10/2022
"""

import motion_env_ascc
import motion_adl_bayes_model


MILAN_BASE_DATE = '2009-10-16'

MILAN_BASE_DATE_HOUR = '2009-10-16 06:00:00'

TEST_BASE_DATE = '2009-12-11'




env = motion_env_ascc.EnvASCC(TEST_BASE_DATE + '00:00:00')
env.reset()

bayes_model_vision = motion_adl_bayes_model.Bayes_Model_Vision()

cur_activity_prob = 0
pre_activity = ''
cur_activity = ''

res_prob = {}
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

    for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():
        p =  bayes_model_vision.prob_of_location_under_act(location, act) \
             * motion_adl_bayes_model.HMM_START_MATRIX[act] /(bayes_model_vision.prob_of_location_under_all_acts(location)) * bayes_model_vision.prob_of_location_using_vision(location, act)
             
        res_prob[act].append(p)

    pre_activity = cur_activity


while(not env.done):

    # INTERVAL_FOR_COLLECTING_DATA
    audio_data, vision_data, motion_data, transition_motion = env.step(motion_env_ascc.MOTION_ACTION)  

    # detect motion
    #if transition_motion:
        # open all sensors
        # new activity
        # Bayes model prob

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


            res_prob[act].append(p1) 

            # p = p1 + p1*p2 + p1*p3
            # todo print out top 3

            # pre_activity should be the acitivty with high probability
            pre_activity = cur_activity

    else:
        # motion data  or audio data
        # new activity
        # Bayes model prob    

        cur_time = env.get_running_time()
        print('Env Running:', cur_time) 
        cur_activity, cur_beginning_activity, cur_end_activity = \
            bayes_model_vision.get_activity_from_dataset_by_time(cur_time)

        location = cur_activity
        for act in motion_adl_bayes_model.PROB_OF_ALL_ACTIVITIES.keys():

            p2 =  bayes_model_motion.get_prob(pre_activity, act, location)
            # p3 = bayes_model_audio.get_prob(pre_activity, cur_activity, location)


            res_prob[act].append(p1) 

            # p = p2 + p2*p3
            # todo print out top 3

            # pre_activity should be the acitivty with high probability
            pre_activity = cur_activity
    


    


    if p > 0.8:
        pass


