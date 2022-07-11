"""
Brief: Bayes model for activity probability, including vision, motion, audio-based model.
Author: Frank
Date: 07/10/2022
"""

cur_activity_prob = 0
pre_activity = ''
cur_activity = ''

if pre_activity == '':
    # open camera
    # detect activity, cur_activity, pre_activity
    # Bayes model


while( not end day):
    motion, audio, transition_motion = Env.step(action=5seconds)

    # detect motion
    if transition_motion:
        # open camera
        # new activity
        # Bayes model prob
    else:
        # motion data  or audio data
        # new activity
        # Bayes model prob
