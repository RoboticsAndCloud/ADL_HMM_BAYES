"""
Brief: We use open-source data set for the simulation environment
@Author: Fei.Liang
@Date: 08/10/2021
Paper proposal:
https://docs.google.com/document/d/1wtd85OB5lnGRIPESCkamNU-O3fMSz_IUwTgspzic8bM/edit#
https://docs.google.com/spreadsheets/d/12XW3PZJMzoQOc3ugGb_I_gqVh8pD2grAcE7QxYFMX0c/edit#gid=0
@Reference:
python_style_rules:
1) https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/#id21
2) https://google.github.io/styleguide/pyguide.html
"""

"""
For action 17, audio + vision, if using duty-cycle, every minute, we trigger the sensors
to collect and send data, 10 hours for standby, it may use extral 144 mAh for data transmitting.
7*124*600/3600 = 144.66666666666666
"""

# from pickle import FALSE
import random
import time
from typing import Dict
import numpy as np
# import tkinter as tk
from PIL import Image

from datetime import datetime
from datetime import timedelta

import constants
import tools_ascc
import time_adl_res_dict


# import pandas as pd
# Given timestamp in string
#time_str = '2009-10-16 00:01:04'
DAY_FORMAT_STR = '%Y-%m-%d'
date_format_str = '%Y-%m-%d %H:%M:%S'
DATE_HOUR_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

TRAIN_RUNNING_TIME_FORMAT = "%H:%M:%S"

DAY_BEGIN_TIME = '07:00:00'
DAY_END_TIME = '20:00:00'
HOUR_TIME_FORMAT = "%H:%M:%S"

DEBUG = False

STATE_TIME_TRANS = 1000*1000*1000*10

MAX_TRIGGER_TIMES = 1500 *2


np.random.seed(1)

UNIT = 100
HEIGHT = 5
WIDTH = 5

ENERGY_STANDBY = 124  # mA   ## 180 WMU?
ENERGY_TX = 220  # mA
ENERGY_RECORDING_MIC = 130  # mA (to be Done from Ricky)


# WMU
# IMAGE_SIZE = 174 #KB
IMAGE_SIZE = 24.86  # KB
IMAGE_COUNT = 10  # 10 Frames
IMAGE_TAKING_TIME_COST = 1/30  # 30 frames

# AUDIO_FILE_SIZE = 50  # KB, Duration: 5 seconds
# AUDIO_RECORDING_TIME_COST = 5  # 5 seconds


AUDIO_FILE_SIZE = 87  # KB, Duration: 1 seconds
AUDIO_RECORDING_TIME_COST = 1  # 1 seconds



ACCELEROMETER_DATA_SIZE = 8  # KB  float 4 bytes,  3-axis,  data rate 100 HZ, recording time 3 seconds,
ACCELEROMETER_DATA_RECORDING_TIME_COST = 2  # 5 seconds (Reference: xxx)

# speed could vary in the ASCC Lab Environment
WIFI_BANDWIDTH_SPEED = 100  # KB/s

# weight of accuracy and energy consumption for reward
# 0.4, accuracy need to be larger than 0.92, vision is good
# 0.38, accuracy need to be larger than 0.90, vision is good
# 0.3 , accuracy need to be larger than 0.7, vision is good
# Weight_acc + Weight_energy = 1
WEIGHT_ACC = 0.4  # default:0.4, 0.9
WEIGHT_ENERGY = 1 - WEIGHT_ACC

BEGINNIG_EVENT_COMPENSATION = 0.3
MIDDLE_EVENT_COMPENSATION = 0.2
END_EVENT_COMPENSATION = 0.25


# PENALTY_FOR_MISSED_EVENT = 50

PENALTY_FOR_MISSED_EVENT_FACTOR = 0.5
ACC_PENALTY_THRESHOLD = 0.85
ACC_PENALTY_FACTOR = 0.5

# For checking, if it checks frequently, it may get some penalty, because it is tired then.
# Take the middle event times as the factor to judge
TIRED_PENALTY = 0.1
TIRED_THRESHOLD = 4 # check 3 



MOTION_TRIGGERRED_ACTION = 1


INTERVAL_FOR_COLLECTING_DATA = 10 # Seconds

#WMU_AUDIO_ACTION = 0
#WMU_VISION_ACTION = 1
WMU_FUSION_ACTION = 0

ROBOT_FUSION_ACTION = 6

WMU_audio = "WMU_audio"
WMU_vision = "WMU_vision"
Nothing = "Nothing"

WMU_fusion = WMU_audio + WMU_vision
WMU_audio_vision = WMU_audio + WMU_vision
#Robot_audio_vision = "Robot_fusion"
Robot_audio = "Robot_audio"
Robot_vision = "Robot_vision"
Robot_audio_vision = Robot_audio + Robot_vision
Robot_audio_WMU_vision = Robot_audio + WMU_vision

#Robot_WMU_audio = Robot_audio_vision + WMU_audio
#Robot_WMU_vision = Robot_audio_vision + WMU_vision
#Robot_WMU_fusion = Robot_audio_vision + WMU_fusion



#RL_ACTION_DICT_0 = {
#    0: WMU_audio,  
#    1: WMU_vision, 
#    2: WMU_fusion,  
#    3: Robot_audio_vision,
#    4: Robot_WMU_audio, # robot and WMU both capture data
#    5: Robot_WMU_vision,
#    6: Robot_WMU_fusion,
#    7: Nothing
#}


MOTION_RECORD_TIME = 2.3 # 2 seconds for motion data collection
# todo: time cost for recognition:  image: 1.15 for 10 images, audio:0.1, motion: 1.14, totally: 2.5
ACTION_INTERVAL_DICT_0 = {
    0: 1,  # "audio",
    1: 1.0/3, #"vision",
    2: 1, # fusion  # multi-thread, audio, vision simultanously
    3: 0, # robot
    4: 0,
    5: 0,
    6: 1, # Robot_WMU_fusion
    7: 0
}

RL_ACTION_DICT = {
    0: WMU_fusion,  
    1: Robot_audio_vision,
    2: Nothing,

    3: WMU_audio,  
    4: WMU_vision,
    5: Robot_audio,
    6: Robot_vision,  
    7: Robot_audio_WMU_vision,
}

ACTION_INTERVAL_DICT = {
    0: 1, # fusion  # multi-thread, audio, vision simultanously
    1: 0, # robot
    2: 0,

    3: 1, # WMU_audio
    4: 0, # WMU_vision 
    5: 0, # Robot_audio
    6: 0, # Robot_vision
    7: 0, # Robot_audio_WMU_vision
}


MAX_LOCATION_CLASS = 6
PRIVACY_LOCATION_LIST_BASIC = [constants.LOCATION_BEDROOM, constants.LOCATION_BATHROOM]
# PRIVACY_LOCATION_LIST_BASIC = [constants.LOCATION_BEDROOM, constants.LOCATION_BATHROOM, constants.LOCATION_LIVINGROOM]
# PRIVACY_LOCATION_LIST_BASIC = [constants.LOCATION_BEDROOM]

PRIVACY_LOCATION_LIST = [constants.LOCATION_BEDROOM, constants.LOCATION_BATHROOM]
# PRIVACY_LOCATION_LIST = [constants.LOCATION_BEDROOM, constants.LOCATION_BATHROOM, constants.LOCATION_KITCHEN]
# PRIVACY_LOCATION_LIST = [constants.LOCATION_BEDROOM, constants.LOCATION_BATHROOM, constants.LOCATION_LIVINGROOM]

# PRIVACY_LOCATION_LIST = [constants.LOCATION_BATHROOM]



PRIVACY_ACTIVITY_LIST = [constants.ACTIVITY_MASTER_BEDROOM, constants.ACTIVITY_MASTER_BATHROOM, constants.ACTIVITY_GUEST_BATHROOM]

ROBOT_CANNOT_REG_ACTIVITY_LIST = [constants.ACTIVITY_LEAVE_HOME]


# action map
# 0 => audio, 1=>vison, 2=>motion, 3=>audio+vision+motion
# ACTION_DICT = {
#     0: "audio",
#     1: "vision",
#     2: "motion",
#     3: "audio_vision_motion"
# }

# 0 => audio, 1=>vison, 2=>audio+vision   motion is with low accuracy
# Action should be with interval
'''
Mic
Camera
Mic+Camera
Interval (Secconds)
0
1
5
10
30
60
'''
# default interval 0 second
# ACTION_DICT = {
#     0: "audio with 0 interval",
#     1: "vision with 0 interval ",
#     2: "audio_vision with 0 interval"
# }

# ACTION_DICT = {
#     0: "audio with 1 interval",  # "audio",
#     1: "vision with 1/3 interval", #"vision",
#     2: "motion with 2 interval", #"motion"
#     3: "audio vision (fusion) with 2+1/3 interval",  # "fusion",


# }




# ACTION_INTERVAL_DICT = {
#     0: 1.1,  # "audio",
#     1: 1.15 + 1.0/3, #"vision",
#     2: 3+1.14, #"motion"
#     3: 1.1+1.15+1.0/3+3+1.14, # fusion
# }


AUDIO_ACTION_LIST = [0]
VISION_ACTION_LIST = [1]
MOTION_ACTION_LIST = [2]
FUSION_ACTION_LIST = [3]
# AUDIO_AND_VISION_ACTION_LIST = [3]




ACTIVITY_LIST = [
    "Desk_Activity",
    "Guest_Bathroom",
    "Kitchen_Activity",
    "Master_Bathroom",
    "Meditate",
    "Watch_TV",
    "Sleep",
    "Read",
    "Bed_to_Toilet",
    "Chores",
    "Dining_Rm_Activity",
    "Eve_Meds",
    "Leave_Home",
    "Morning_Meds",
    "Master_Bedroom_Activity",
    "Phone_Call",
    "Fall",
    "Medication"]


ACTIVITY_DICT = {
    0: "Desk_Activity",
    1: "Guest_Bathroom",
    2: "Kitchen_Activity",
    3: "Master_Bathroom",
    4: "Meditate",
    5: "Watch_TV",
    6: "Sleep",
    7: "Read",
    8: "Bed_to_Toilet",
    9: "Chores",
    10: "Dining_Rm_Activity",
    11: "Eve_Meds",
    12: "Leave_Home",
    13: "Morning_Meds",
    14: "Master_Bedroom_Activity",
    15: "Phone_Call",
    16: "Fall",
    17: "Medication"
}

ACTIVITY_PRIORITY_DICT = {
    0: 2,
    1: 2,
    2: 2,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 3,
    9: 3,
    10: 3,
    11: 3,
    12: 3,
    13: 3,
    14: 3,
    15: 3,
    16: 1,
    17: 1
}

# accuracy for begin, middle, end
AUDIO_ACTIVITY_ACCURACY_DICT = {
    0: [0.731, 0.731, 0.731],
    1: [0.55, 0.667, 0.667],
    2: [0.92, 0.92, 0.99],
    3: [0.55, 0.667, 0.667],
    4: [0.925, 0.925, 0.925],
    5: [0.709, 0.709, 0.709],  # watch TV for audio should be with higher accuracy
    6: [0.55, 0.971, 0.56],
    7: [0.01, 0.01, 0.01],
    8: [0.55, 0.667, 0.683],
    9: [0.887, 0.887, 0.887],
    10: [0.536, 0.804, 0.669],
    11: [0.925, 0.925, 0.925],
    12: [0.55, 0.01, 0.55],
    13: [0.925, 0.925, 0.925],
    14: [0.55, 0.55, 0.55],
    15: [0.787, 0.787, 0.787],
    16: [0.693, 0.693, 0.693],
    17: [0.73, 0.65, 0.73]
}

VISION_ACTIVITY_ACCURACY_DICT = {
    0: [0.983, 0.983, 0.983],
    1: [0.8169, 0.8169, 0.8169],
    2: [0.91, 0.91, 0.91],
    3: [0.8169, 0.8169, 0.8169],
    4: [0.75, 0.75, 0.75],
    5: [0.95, 0.5556, 0.95],
    6: [0.93, 0.3, 0.93],
    7: [0.8647, 0.8647, 0.8647], # changed
    8: [0.8169, 0.8169, 0.8169],
    9: [0.8169, 0.8169, 0.8169],
    10: [0.8169, 0.5625, 0.8169],
    11: [0.75, 0.75, 0.75],
    12: [0.78, 0.01, 0.78],
    13: [0.75, 0.75, 0.75],
    14: [0.93, 0.93, 0.93],
    15: [0.01, 0.01, 0.01],
    16: [0.94, 0.94, 0.94],
    17: [0.92, 0.92, 0.92]
}

MOTION_ACTIVITY_ACCURACY_DICT = {
    0: [0.01, 0.01, 0.01],
    1: [0.01, 0.01, 0.01],
    2: [0.01, 0.01, 0.01],
    3: [0.01, 0.01, 0.01],
    4: [0.01, 0.01, 0.01],
    5: [0.01, 0.01, 0.01],
    6: [0.99, 0.01, 0.99],
    7: [0.01, 0.01, 0.01],
    8: [0.01, 0.01, 0.01],
    9: [0.01, 0.01, 0.01],
    10: [0.01, 0.01, 0.01],
    11: [0.01, 0.01, 0.01],
    12: [0.01, 0.01, 0.01],
    13: [0.01, 0.01, 0.01],
    14: [0.01, 0.01, 0.01],
    15: [0.01, 0.01, 0.01],
    16: [0.01, 0.01, 0.01],
    17: [0.01, 0.01, 0.01]
}


FUSION_ACTIVITY_ACCURACY_DICT = {
    0: [0.99, 0.99, 0.99],
    1: [0.99, 0.99, 0.99],
    2: [0.99, 0.99, 0.99],
    3: [0.99, 0.99, 0.99],
    4: [0.99, 0.99, 0.99],
    5: [0.99, 0.99, 0.99],
    6: [0.99, 0.99, 0.99],
    7: [0.99, 0.99, 0.99],
    8: [0.99, 0.99, 0.99],
    9: [0.99, 0.99, 0.99],
    10: [0.99, 0.99, 0.99],
    11: [0.99, 0.99, 0.99],
    12: [0.99, 0.99, 0.99],
    13: [0.99, 0.99, 0.99],
    14: [0.99, 0.99, 0.99],
    15: [0.99, 0.99, 0.99],
    16: [0.99, 0.99, 0.99],
    17: [0.99, 0.99, 0.99]
}


# 1200 600 300 100
# Date: 2010-01-05 Begin: 08:08:39 End: 22:58:17 Duration(Hours):.2f 14.82
# BATTERY_LIFE = 1500 * 3600  # mAh * 3600
#BATTERY_LIFE = 2000 * 3600  # mAh * 3600
BATTERY_LIFE = 5000 * 3600  # mAh * 3600


EXPECTED_TOLERANCE = 1  # tolerance in seconds, if the predict activity time is around $EXPECTED_TOLERANCE, the predict activity is good

ROBOT_WAIT_DURATION = 20 # wait the robot to move to the target location due to the speed of the moblie base

def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

class EnvASCC():
    def __init__(self, time_str = '2009-12-11 08:00:00'):
        
        high = np.array([1.0, 1.0, BATTERY_LIFE], dtype=np.float32)

        #self.n_actions = 0.30
        self.n_actions = 0.42

        self.n_features = 2

        self.texts = []

        self.residual_power = BATTERY_LIFE
        self.date_time_str = time_str  # simulate the activity in this day
        self.running_time = datetime.strptime(self.date_time_str, DATE_HOUR_TIME_FORMAT)
        self.running_day = datetime.strptime(self.date_time_str.split()[0], DAY_FORMAT_STR)

        self.activity = 0 # activity map
        
        self.activity_date_dict = {}
        self.activity_begin_dict = {}
        self.activity_end_dict = {}
        # activity_begin_dict = Dictlist()
        # activity_end_dict = Dictlist()
        self.activity_begin_list = []
        self.activity_end_list = []

        self.activity_dict_init()

        self.day_begin, self.day_end = tools_ascc.get_day_begin_end(self.activity_date_dict, 
                                            self.activity_begin_dict, self.activity_end_dict)

        self.total_activity_cnt = tools_ascc.get_activity_count_by_date(self.date_time_str.split()[0]) -2 # ignore two sleep activities
                                            
        self.initial_state = 6 # sleep

        # keep incresing until the max_steps, we need this for the code
        self.missed_event_times = 0
        self.hit_event_times = 0
        self.totol_check_times = 0
        self.uncertain_times = 0

        self.activity_none_times = 0
        self.expected_activity_none_times = 0

        self.activity_counter_dict = {}
        self.beginning_event_times = 0
        self.middle_event_times = 0
        self.end_event_times = 0
        self.activity_middle_event_times = 0

        self.random_miss_event_times = 0

        self.missed_expected_known_event_times = 0


        self.energy_cost = 0
        self.time_cost = 0

        self.hit_activity_check_times = 0
        self.miss_activity_check_times = 0

        self.motion_activity_cnt = 0
        self.need_recollect_data_cnt = 0




        self.done_total_time_cost = 0

        self.done = False
        self.done_reward = 0
        self.done_residual_power = self.residual_power
        self.done_running_time = self.running_time

        # when done == True, that means either the power is off or day is end, the following 
        # factores remain the same
        self.done_missed_event_times = 0
        self.done_hit_event_times = 0
        self.done_totol_check_times = 0
        self.done_uncertain_times = 0
        self.done_activity_counter_dict = {}
        self.done_beginning_event_times = 0
        self.done_middle_event_times = 0
        self.done_end_event_times = 0

        self.done_random_miss_event_times = 0
        self.done_penalty_times = 0

        self.done_energy_cost = 0
        self.done_time_cost = 0



        # todo self.res_hit_event_dict{}
        self.res_hit_event_dict = {}
        self.res_miss_event_dict = {}
        self.res_real_miss_event_dict = {}

        self.res_random_event_dict = {}

        self.res_hit_plus_miss_event_dict = {}


        self.motion_triggered_interval = 0
        self.motion_triggered_times = 0

        self.wmu_mic_times = 0
        self.wmu_cam_times = 0
        self.wmu_times = 0
        self.robot_mic_trigger_times = 0
        self.robot_cam_trigger_times = 0
        self.robot_trigger_times = 0
        self.privacy_occur_cnt = 0
        # Reset the running time to day_begin
        self.reset()



    def activity_dict_init(self):
        day_hour_format_str = self.running_time.strftime(DATE_HOUR_TIME_FORMAT)
        day_str = day_hour_format_str.split()[0]

        self.activity_date_dict, self.activity_begin_dict, self.activity_end_dict, \
        self.activity_begin_list, self.activity_end_list  = tools_ascc.get_activity_date(day_str)

    def display_info(self):

        print("Date:", self.running_day, " Begin:", self.day_begin.strftime(HOUR_TIME_FORMAT), 
        " End:", self.day_end.strftime(HOUR_TIME_FORMAT),
              " Duration(Hours):", (self.day_end - self.day_begin).seconds/3600.0)
        print("Initial Battery Life:", BATTERY_LIFE)

    def reset(self):
        self.residual_power = BATTERY_LIFE

        day_hour_time = self.running_day.strftime(DAY_FORMAT_STR) + " " \
             + self.day_begin.strftime(HOUR_TIME_FORMAT) 
        self.running_time = datetime.strptime(day_hour_time, DATE_HOUR_TIME_FORMAT)

        train_running_time = self.get_train_running_time()

        self.done = False
        self.done_reward = 0
        self.done_residual_power = self.residual_power
        self.done_running_time = self.running_time
        self.done_missed_event_times = 0
        self.done_hit_event_times = 0
        self.done_totol_check_times = 0
        self.done_uncertain_times = 0
        self.done_beginning_event_times = 0
        self.done_middle_event_times = 0
        self.done_end_event_times = 0
        self.done_random_miss_event_times = 0
        self.done_penalty_times = 0
        self.done_energy_cost = 0
        self.done_time_cost = 0
        self.missed_expected_known_event_times = 0


        self.hit_activity_check_times = 0
        self.miss_activity_check_times = 0

        self.motion_activity_cnt = 0


        self.activity_none_times = 0
        self.expected_activity_none_times = 0
        self.totol_check_times = 0
        self.missed_event_times = 0
        self.hit_event_times = 0
        self.uncertain_times = 0
        self.beginning_event_times = 0
        self.middle_event_times = 0
        self.end_event_times = 0
        self.random_miss_event_times = 0
        self.activity_middle_event_times = 0
        self.energy_cost = 0
        self.time_cost = 0
        self.res_real_miss_event_dict = {}
        self.done_total_time_cost = 0
        self.res_random_event_dict = {}
        self.res_hit_event_dict = {}
        self.res_hit_plus_miss_event_dict = {}

        self.runing_time_action_dict = {}
        self.runing_time_action_dict_motion = {}

        self.motion_triggered_times = 0

        self.wmu_mic_times = 0
        self.wmu_cam_times = 0
        self.wmu_times = 0
        self.robot_mic_trigger_times = 0
        self.robot_cam_trigger_times = 0
        self.robot_trigger_times = 0
        self.privacy_occur_cnt = 0
        self.privacy_occur_cnt_level1_robot_vision = 0
        self.privacy_occur_cnt_level1_robot_sound = 0
        self.privacy_occur_cnt_level1_wmu_sound = 0

        self.privacy_occur_cnt_level2_robot_vision = 0
        self.privacy_occur_cnt_level2_robot_sound = 0
        self.privacy_occur_cnt_level2_wmu_sound = 0


        next_state_ = [train_running_time / STATE_TIME_TRANS, (self.activity / 100)]
        # next_state_ = [train_running_time, self.initial_state, self.residual_power]
        # todo change it to real number

        # For debug purpos
        self.running_time = datetime.strptime('2009-12-11 08:46:27', DATE_HOUR_TIME_FORMAT)

        return next_state_

    def step(self, p_action, need_recollect_data = False):
        
        if DEBUG:
            print("Step--running_time:", self.running_time)

        if not self.done:
            self.done_totol_check_times = self.done_totol_check_times + 1
        
        self.totol_check_times = self.totol_check_times + 1



        action = p_action
        # if(p_action[0] in {'NaN','infinity','nan','Infinity'}):
        #     action = 0
        
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("CAMS-DQN: Action:", RL_ACTION_DICT[p_action])
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        # import adl_env_client_lib
        # import adl_type_constants
        # try:
        #     if RL_ACTION_DICT[p_action] == WMU_fusion or RL_ACTION_DICT[p_action] == Robot_audio_vision or RL_ACTION_DICT[p_action] == Robot_audio_WMU_vision:

        #         adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_IPRECEIVE, adl_type_constants.WMU_RECEIVE_PORT,
        #                                                     adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_FUSION)

        #         # adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_IPRECEIVE, adl_type_constants.WMU_RECEIVE_PORT,
        #         #                                             adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_AUDIO)
                
        #         # adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_COMPANION_ROBOT_IPRECEIVE, adl_type_constants.WMU_COMPANION_ROBOT_RECEIVE_PORT,
        #         #                                             adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_FUSION)
                
        #         # send cmd to the robots
        #         if RL_ACTION_DICT[p_action] == Robot_audio_vision:
        #             time.sleep(ROBOT_WAIT_DURATION)
        #             adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_COMPANION_ROBOT_IPRECEIVE, adl_type_constants.WMU_COMPANION_ROBOT_RECEIVE_PORT,
        #                                                     adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_FUSION)
        #         #self.fusion_check_times += 1
        #         # adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_COMPANION_ROBOT_IPRECEIVE, adl_type_constants.WMU_COMPANION_ROBOT_RECEIVE_PORT,
        #         #                                             adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_AUDIO)
        #         # adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_COMPANION_ROBOT_IPRECEIVE, adl_type_constants.WMU_COMPANION_ROBOT_RECEIVE_PORT,
        #         #                                             adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_IMAGE)

        #     elif RL_ACTION_DICT[p_action] == Nothing:
        #         adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_IPRECEIVE, adl_type_constants.WMU_RECEIVE_PORT,
        #                                                         adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_MOTION)
        #         #self.motion_check_times += 1
        #     elif RL_ACTION_DICT[p_action] == WMU_vision or RL_ACTION_DICT[p_action] == Robot_vision:
        #         if RL_ACTION_DICT[p_action] == WMU_vision:
        #             adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_IPRECEIVE, adl_type_constants.WMU_RECEIVE_PORT,
        #                                                     adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_IMAGE)
                
        #         # send cmd to the robots
        #         if RL_ACTION_DICT[p_action] == Robot_audio_vision or RL_ACTION_DICT[p_action] == Robot_vision:
        #             time.sleep(ROBOT_WAIT_DURATION)
        #             adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_COMPANION_ROBOT_IPRECEIVE, adl_type_constants.WMU_COMPANION_ROBOT_RECEIVE_PORT,
        #                                                     adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_IMAGE)
    
        #     elif RL_ACTION_DICT[p_action] == WMU_audio or RL_ACTION_DICT[p_action] == Robot_audio:
        #         adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_IPRECEIVE, adl_type_constants.WMU_RECEIVE_PORT,
        #                                                     adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_AUDIO)
                
        #         # send cmd to the robots
        #         if RL_ACTION_DICT[p_action] == Robot_audio:
        #             time.sleep(ROBOT_WAIT_DURATION)
        #             adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_COMPANION_ROBOT_IPRECEIVE, adl_type_constants.WMU_COMPANION_ROBOT_RECEIVE_PORT,
        #                                                     adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_AUDIO)
        
        # except Exception as e:
        #     print("Got error when Send cmd to WMU, err:", e, " p_action:", p_action, ' ip2:', adl_type_constants.WMU_IPRECEIVE)
        #     #logging.warn('Got error when Send cmd to WMU')
        #     #logging.warn(e)
        #     # return # the watch app may crush
        #     return '', '', '', False

        
        

        running_time_str = self.running_time.strftime(TRAIN_RUNNING_TIME_FORMAT)

        running_time_str = self.running_time.strftime(TRAIN_RUNNING_TIME_FORMAT)
        self.runing_time_action_dict[running_time_str] = action

        interval = ACTION_INTERVAL_DICT[action]

        ###############################change here ###########################


        # interval = interval - 3 + 60*2 # 30 seconds, 1 min

        pre_interval = interval

        pre_action = action

        print("Running time:", self.running_time, "Action:", action)
        

        ###############################change here ###########################

        action, self.motion_triggered_interval = pre_action, pre_interval

        motion_triggerred_flag = False

        motion_interval = 3 # 3 seconds to detect the motion activity
        # if action == MOTION_ACTION or action == FUSION_ACTION:
        #     motion_triggerred_flag, self.motion_triggered_interval = self.check_motion_action(action, motion_interval)


        self.runing_time_action_dict_motion[running_time_str] = action

        self.action_count_store(action)

        # sensors_power_consumption, sensors_time_cost = self.sensors_energy_time_cost(action)

        # if DEBUG:
        #     print("Step--p_action power_consumption:", sensors_power_consumption, " time cost:", sensors_time_cost)

        done = False

        # power restriction
        # self.residual_power = self.residual_power - sensors_power_consumption


        # self.time_cost = self.time_cost + sensors_time_cost 
        # self.energy_cost = self.energy_cost + sensors_power_consumption

        self.running_time = self.get_current_running_time(interval+ MOTION_RECORD_TIME)


        if not self.done:
            # self.done_reward = reward
            self.done_residual_power = self.residual_power
            self.done_running_time = self.running_time
            # self.done_time_cost = self.done_time_cost + sensors_time_cost
            # self.done_energy_cost = self.done_energy_cost + sensors_power_consumption
            # self.done_total_time_cost = self.done_total_time_cost + sensors_time_cost

        if self.wmu_cam_times > MAX_TRIGGER_TIMES or self.robot_cam_trigger_times > MAX_TRIGGER_TIMES:
            done = True #  Battery is dead
            self.done = True
            # print("Done = True, self.residual_power: ", self.residual_power)
        


        if self.residual_power <= 0:
            done = True #  Battery is dead
            self.done = True
            # print("Done = True, self.residual_power: ", self.residual_power)
        

        # self.running_time = self.get_current_running_time(sensors_time_cost)

        if not self.done:
            self.done_running_time = self.running_time

        if self.get_current_hour_time() > self.day_end:
            if DEBUG:
                print("self.running_time:", self.running_time, "self.day_end:", self.day_end)
            done = True
            self.done = True
            # print("Done == True, self.running_time:", self.running_time, "self.day_end:", self.day_end)
        
        #print('Get current date_day_time:', self.get_current_date_day_time(), " running day:", self.running_day)


        if self.get_current_date_day_time() > self.running_day:
            print('Get current date_day_time:', self.get_current_date_day_time(), " running day:", self.running_day)
            done = True
            self.done = True

        if self.done:
            if DEBUG:
                print("Step Done || Running time: ", self.running_time, "Residual power: ", self.residual_power)


        self.res_hit_plus_miss_event_dict = merge_dicts(self.res_hit_event_dict, self.res_random_event_dict)


        if need_recollect_data == False:
            if WMU_audio in RL_ACTION_DICT[action]:
                self.wmu_mic_times += 1
            if WMU_vision in RL_ACTION_DICT[action]:
                self.wmu_cam_times += 1
            if WMU_audio_vision in RL_ACTION_DICT[action]:
                self.wmu_times += 1

            if Robot_audio in RL_ACTION_DICT[action]:
                self.robot_mic_trigger_times += 1
            if Robot_vision in RL_ACTION_DICT[action]:
                self.robot_cam_trigger_times += 1
            if Robot_audio_vision in RL_ACTION_DICT[action]:
                self.robot_trigger_times += 1
        else:
            self.need_recollect_data_cnt += 1

            if WMU_audio in RL_ACTION_DICT[action]:
                self.wmu_mic_times += 1
            if WMU_vision in RL_ACTION_DICT[action]:
                self.wmu_cam_times += 1
            if WMU_audio_vision in RL_ACTION_DICT[action]:
                self.wmu_times += 1

            if Robot_audio in RL_ACTION_DICT[action]:
                self.robot_mic_trigger_times += 1
            if Robot_vision in RL_ACTION_DICT[action]:
                self.robot_cam_trigger_times += 1
            if Robot_audio_vision in RL_ACTION_DICT[action]:
                self.robot_trigger_times += 1

            
        return '', '', '', motion_triggerred_flag
    
    
    def get_wmu_sensor_trigger_times(self):

        return self.wmu_mic_times, self.wmu_cam_times

    def get_reward_energy(self, action):
        reward = 0
        if WMU_audio in RL_ACTION_DICT[action]:
             reward = reward + 1.28
        if WMU_vision in RL_ACTION_DICT[action]:
             reward = reward + 1
             #battery_level = self.get_battery_level()
             #reward = reward + battery_level * 0.1
        
        if self.wmu_cam_times > MAX_TRIGGER_TIMES * 0.5:
            reward = reward * (1+ self.wmu_cam_times*1.0/MAX_TRIGGER_TIMES)

        # if Robot_audio in RL_ACTION_DICT[action]:
        #      reward = reward + 0.2
        # if Robot_vision in RL_ACTION_DICT[action]:
        #      reward = reward + 0.5

        return reward
    
    # def get_reward_privacy(self, action, time_str):

    #     reward = 0
    #     target_folder_time_str = tools_ascc.get_target_folder_time_str(time_str)
    #     location = ''

    #     if target_folder_time_str in time_adl_res_dict.time_location_dict.keys():
    #         location, prob = time_adl_res_dict.time_location_dict[target_folder_time_str]
    #     else:
    #         location, prob = tools_ascc.get_activity_by_vision_dnn(time_str, action='vision')
    #         print('get_location_by_activity_CNN time_str:', time_str, ' location:', location, ' prob:', prob)


    #     if location in PRIVACY_LOCATION_LIST:
    #         if Robot_audio_vision in RL_ACTION_DICT[action]:
    #             reward = reward + 1
    #             self.privacy_occur_cnt = self.privacy_occur_cnt + 1

    #     return reward

    def get_reward_accuracy(self, action, activity):

        acc = 1

        if activity in ROBOT_CANNOT_REG_ACTIVITY_LIST:
            if Robot_audio_vision == RL_ACTION_DICT[action]:

                acc = -1

        return acc

    def get_reward_privacy(self, action, activity, location=''):

        reward = 0
        print('location checking:', location)
        level1 = 0.8
        level2 = 0.2

        vision_p = 0.8
        audio_p = 0.2

        # level 1, 0.8

        if location != '' and location in PRIVACY_LOCATION_LIST:
            if Robot_vision in RL_ACTION_DICT[action]:
                reward = reward + level1 * vision_p
                self.privacy_occur_cnt = self.privacy_occur_cnt + 1
                self.privacy_occur_cnt_level1_robot_vision = self.privacy_occur_cnt_level1_robot_vision + 1
                print('location !=', location)
            if Robot_audio in RL_ACTION_DICT[action]:
                reward = reward + level1 * audio_p
                self.privacy_occur_cnt = self.privacy_occur_cnt + 1
                self.privacy_occur_cnt_level1_robot_sound = self.privacy_occur_cnt_level1_robot_sound + 1
                print('location !=', location)
            if WMU_audio in RL_ACTION_DICT[action]:
                reward = reward + level1 * audio_p
                self.privacy_occur_cnt = self.privacy_occur_cnt + 1
                self.privacy_occur_cnt_level1_wmu_sound = self.privacy_occur_cnt_level1_wmu_sound + 1
                print('location !=', location)            
        else:
            # level 2
            if Robot_vision in RL_ACTION_DICT[action]:
                reward = reward + level2 * vision_p
                self.privacy_occur_cnt = self.privacy_occur_cnt + 1
                self.privacy_occur_cnt_level2_robot_vision = self.privacy_occur_cnt_level2_robot_vision + 1
                print('location !=', location)
            if Robot_audio in RL_ACTION_DICT[action]:
                reward = reward + level2 * audio_p
                self.privacy_occur_cnt = self.privacy_occur_cnt + 1
                self.privacy_occur_cnt_level2_robot_sound = self.privacy_occur_cnt_level2_robot_sound + 1
                print('location !=', location)
            if WMU_audio in RL_ACTION_DICT[action]:
                reward = reward + level1 * audio_p
                self.privacy_occur_cnt = self.privacy_occur_cnt + 1
                self.privacy_occur_cnt_level2_wmu_sound = self.privacy_occur_cnt_level2_wmu_sound + 1
                print('location !=', location)   

        return reward

        if activity in PRIVACY_ACTIVITY_LIST:
            if Robot_vision in RL_ACTION_DICT[action]:
                reward = reward + 1
                self.privacy_occur_cnt = self.privacy_occur_cnt + 1
            #else:
            #    reward = -1 # get higher reward for using wmu sensors to detect private activity

        return reward
    
    #
    def render(self):
        # time.sleep(0.03)
        self.update()

    def get_priority(self, p):
        n_p = 1
        if p == 1:
            n_p = 1
        elif p == 2:
            n_p = 0.5
        elif p == 3:
            n_p = 0.1

        return n_p

    def action_count_store(self, action):

        if action in self.activity_counter_dict:
            self.activity_counter_dict[action] = self.activity_counter_dict[action] + 1
        else:
            self.activity_counter_dict[action] = 1

        if not self.done:
            if action in self.done_activity_counter_dict:
                self.done_activity_counter_dict[action] = self.done_activity_counter_dict[action] + 1
            else:
                self.done_activity_counter_dict[action] = 1
    
    def display_action_counter(self):
        print("====Action, Times====")
        for i in range(len(ACTION_DICT)):
            if i in self.activity_counter_dict:
                print("Action:", i, self.activity_counter_dict[i])
        print("====Action, Times====")
        for i in range(len(ACTION_DICT)):
            if i in self.done_activity_counter_dict:
                print("Action(Done):", i, self.done_activity_counter_dict[i])

        print("====Running time, Action ====")
        for i in self.runing_time_action_dict:
            print("Action(Done):\t", i, '\t', self.runing_time_action_dict[i])

        print("====Running time, Action Motion triggered====")
        for i in self.runing_time_action_dict_motion:
            print("Action(Done):\t", i, '\t', self.runing_time_action_dict_motion[i])       
        
        
        print("====Hit Event Dict: Time, Activity====, total:", len(self.res_hit_event_dict))
        print("Hit \t time \t activity_index \t activity")
        for i in self.res_hit_event_dict:
            if self.res_hit_event_dict[i] >= 0:
                print("Hit:", i, self.res_hit_event_dict[i], ACTIVITY_DICT[self.res_hit_event_dict[i]])
            else:
                print("Hit:", i, self.res_hit_event_dict[i])

        print("====Random Event Dict: Time, Activity====, total:", len(self.res_random_event_dict))
        for i in self.res_random_event_dict:
            if self.res_random_event_dict[i] >= 0:
                print("Random Miss:", i, self.res_random_event_dict[i], ACTIVITY_DICT[self.res_random_event_dict[i]])
            else:
                print("Random Miss:", i, self.res_random_event_dict[i])

        print("====Hit + Random Event Dict: Time, Activity====, total:", len(self.res_hit_plus_miss_event_dict))
        for i in self.res_hit_plus_miss_event_dict:
            if self.res_hit_plus_miss_event_dict[i] >= 0:
                print("Hit + Miss:", i, self.res_hit_plus_miss_event_dict[i], ACTIVITY_DICT[self.res_hit_plus_miss_event_dict[i]])
            else:
                print("Hit + Miss:", i, self.res_hit_plus_miss_event_dict[i])

        print("========================get_activity_duration_output========================")
        self.get_activity_duration_for_day_output()
        print("========================get_activity_duration_output========================")

        print("==============get_activity_duration_output_by_hit_miss_event================")
        self.get_activity_duration_for_day_output_by_hit_event()
        print("==============get_activity_duration_output_by_hit_miss_event================")

        #
        # print("====Miss Event Dict: Time, Activity====, total:", len(self.res_miss_event_dict))
        # for i in self.res_miss_event_dict:
        #     if self.res_miss_event_dict[i] >= 0:
        #         print("Miss:", i, self.res_miss_event_dict[i], ACTIVITY_DICT[self.res_miss_event_dict[i]])
        #     else:
        #         print("Miss:", i, self.res_miss_event_dict[i])
        #
        # print("====Miss(Real) Event Dict: Time, Activity====, total:", len(self.res_real_miss_event_dict))
        # for i in self.res_real_miss_event_dict:
        #     if self.res_real_miss_event_dict[i] >= 0:
        #         print("Miss:", i, self.res_real_miss_event_dict[i], ACTIVITY_DICT[self.res_real_miss_event_dict[i]])
        #     else:
        #         print("Miss:", i, self.res_real_miss_event_dict[i])

    def get_activity_duration_for_day_output_by_hit_event(self):
        hit_event_dict = self.res_hit_plus_miss_event_dict
        sd_hit_event_dict = sorted(hit_event_dict.items())

        # ignore the unsure activity : -1
        last_hit_act = -100
        time_list = []
        a_begin = self.day_begin
        a_end = self.day_end

        output_dict = {}
        output_dict2 = {}

        key = ''

        print("tttttttttttttttttttttttttttt:")
        print(sd_hit_event_dict)

        for k_time, v_act in sd_hit_event_dict:
            # if sd_hit_event_dict[i] < 0:
            #     continue
            if (last_hit_act != v_act):
                if last_hit_act != -100 and last_hit_act >= 0:
                    # output

                    # If just detect the activity once, that means we have only a_begin, not a_end
                    # Two cases: 1) without end time 2) using the beginning time of the next activity as end time
                    if len(time_list) == 1:
                        a_end = datetime.strptime(k_time, DATE_HOUR_TIME_FORMAT)

                    duration = (a_end - a_begin).seconds * 1.0 /60
                    t_str = ''
                    for t in time_list:
                        t_str = t_str + t + '\t'

                    tmp_str = key + '\t' + t_str + str(duration)
                    tmp_str2 = t_str + key + '\t' + str(duration)
                    output_dict[a_begin] = tmp_str
                    output_dict2[a_begin] = tmp_str2


                last_hit_act = v_act
                if last_hit_act > 0:
                    key =  ACTIVITY_DICT[last_hit_act]

                a_begin = datetime.strptime(k_time, DATE_HOUR_TIME_FORMAT)
                time_list = []
                time_list.append(k_time)
            
            else:
                time_list.append(k_time)
                a_end = datetime.strptime(k_time, DATE_HOUR_TIME_FORMAT)

        sd = sorted(output_dict.items())
        sd2 = sorted(output_dict2.items())        
        print("Hit + Miss Event For Activity Monitoring")
        print("Activity \t Hit time \t Duration")
        for k,v in sd:
            print(v)

        print('=================')
        print("Hit time \t Activity \t Duration")
        for k,v in sd2:
            print(v)
        print("End len of the activity:", len(output_dict))


            

        

    def get_activity_duration_for_day_output(self):

        day_time_str = self.running_day
        activity_begin_dict = self.activity_begin_dict
        activity_end_dict = self.activity_end_dict

        print("=====================================")
        print("Date:", day_time_str)
        print("activity_begin_dict", len(activity_begin_dict))
        print("activity_end_dict", len(activity_end_dict))

        motion_activity_cnt = 0
        import collections
        output_dict = {}
        output_dict2 = {}  # timestamp is at the beginning

        output_miss_event_dict = {}

        hit_event_dict = self.res_hit_event_dict
        sd_hit_event_dict = sorted(hit_event_dict.items())


        last_hit_time = self.day_begin
        #  5+0.5+2.48+1/3 = > 8.313333333333334
        _, sensors_time_cost = self.sensors_energy_time_cost(38)

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
                a_begin = datetime.strptime(time_list_begin[t_i], DATE_HOUR_TIME_FORMAT)
                try:
                    a_end = datetime.strptime(time_list_end[t_i], DATE_HOUR_TIME_FORMAT)
                except:
                    print("End list not good", len(time_list_begin), len(time_list_end))
                    break

                duration = (a_end - a_begin).seconds * 1.0 /60


                # each day start after getting up (sleep end), ignore the activies before the time of 'sleep end'
                tmp_a_end = datetime.strptime(time_list_begin[t_i].split()[1], HOUR_TIME_FORMAT)
                if tmp_a_end < self.day_begin:
                    print("A end < day begin, ignore:", a_end, self.day_begin)
                    break

                motion_activity_cnt = motion_activity_cnt + 1

                if duration > 0:
                    tmp_str = key + "\t" + time_list_begin[t_i] + "\t" + time_list_end[t_i] + "\t" + str(duration)
                    tmp_str2 = time_list_begin[t_i] + "\t" + time_list_end[t_i] + "\t" + key + "\t" + str(duration)

                    output_dict[time_list_begin[t_i]] = tmp_str
                    output_dict2[time_list_begin[t_i]] = tmp_str2


                    for k_time, v in sd_hit_event_dict:
                        # print("kkkkk:",k_time, "  vvvvv",v)
                        if v < 0:
                           continue
                        hit_time = datetime.strptime(k_time, DATE_HOUR_TIME_FORMAT)
                        

                        # todo check if hit_time + sensors_time_cost , if it meet the acitivity time, then also considerred as hti

                        if ACTIVITY_DICT[v] == key:
                            if hit_time >= a_begin and hit_time <= a_end:
                                self.hit_activity_check_times = self.hit_activity_check_times + 1
                                ## Note: in dict, the time is out off order
                                last_hit_time_list.append(hit_time)
                                # print("#####:hit time:", hit_time, key)
                                break
                        else:

                            # no_acitivity_during_data_collection = True

                            # for last_hit_time in last_hit_time_list:
                            #     print("tttest: last_hit_time:", last_hit_time, "a_begin:", a_begin)

                            #     if last_hit_time > a_begin or last_hit_time < (a_begin - timedelta(seconds=20)):
                            #         continue

                            #     test_acc_during_data_collection = last_hit_time 
                            #     no_acitivity_during_data_collection = True

                            #     # here, sensor time = 8.333, for the range statement,, need to be int(8.3+2), 0 1 2 3 4 5 6 7 8 9, then hit time format we use int(running time), need extra 0.5
                            #     for i in range (int(sensors_time_cost + 3)):
                            #         print("## i:", i, "test_acc_during_data_collection:", test_acc_during_data_collection, "key:", key, "a_begin:", a_begin, "a_end:", a_end)
                            #         target_time = test_acc_during_data_collection + timedelta(seconds=i)
                            #         if target_time >= a_begin  \
                            #                 and  target_time <= a_end:

                            #             self.hit_activity_check_times = self.hit_activity_check_times + 1

                            #             no_acitivity_during_data_collection = False
                            #             break

                            if hit_time > a_end:                        
                                #self.miss_activity_check_times = self.miss_activity_check_times + 1
                                output_miss_event_dict[time_list_begin[t_i]] = tmp_str
                                last_miss_time_set.add(a_begin)

                    # print(key, "\t", time_list_begin[t_i], "\t", time_list_end[t_i], "\t", duration)

        # if len(activity_begin_list) == 0:
        #     continue

        sd = sorted(output_dict.items())
        sd2 = sorted(output_dict2.items())

        for k,v in sd:
            print(v)
        
        print("---------------------------------------------------------")
        for k,v in sd2:
            print(v)
        
        self.motion_activity_cnt = motion_activity_cnt
        print("##########################################")
        print("Total_activity_cnt", self.total_activity_cnt)
        print("motion_activity_cnt:", motion_activity_cnt)
        print("Hit_activity_check_times:",  self.hit_activity_check_times)
        print("Miss_activity_check_times:",  len(output_miss_event_dict))
        print("Miss activity list:")
        miss_activity_ratio =  len(output_miss_event_dict) * 1.0 / (self.total_activity_cnt)
        print("Miss activity ratio:", miss_activity_ratio)

        activity_detection_ratio = self.hit_activity_check_times * 1.0 /self.total_activity_cnt
        print("Activity detection ratio:", activity_detection_ratio)

        sd = sorted(output_miss_event_dict.items())
        for k,v in sd:
            print(v)

        
        # try to detect the events occurs during the data collection using sensors
        # print("len(miss, hit):", len(last_miss_time_list), len(last_hit_time_list))
        # print(last_miss_time_list)
        for miss_a_begin in last_miss_time_set:
            a_begin = miss_a_begin

            for last_hit_time in last_hit_time_list:

                if last_hit_time > a_begin or last_hit_time < (a_begin - timedelta(seconds=20)):
                    continue

                test_acc_during_data_collection = last_hit_time 

                # here, sensor time = 8.333, for the range statement,, need to be int(8.3+2), 0 1 2 3 4 5 6 7 8 9, then hit time format we use int(running time), need extra 0.5
                for i in range (int(sensors_time_cost + 3)):
                    #print("## i:", i, "test_acc_during_data_collection:", test_acc_during_data_collection, "key:", key, "a_begin:", a_begin, "a_end:", a_end)
                    target_time = test_acc_during_data_collection + timedelta(seconds=i)
                    if target_time >= a_begin  \
                            and  target_time <= a_end:

                        self.hit_activity_check_times = self.hit_activity_check_times + 1
                        self.miss_activity_check_times = self.miss_activity_check_times - 1
                        print("Found the event during the data collation:", a_begin)
                        break

        print("##########################################")
        print("motion_activity_cnt:", motion_activity_cnt)
        print("Hit_activity_check_times:",  self.hit_activity_check_times)
        print("Miss_activity_check_times:",  motion_activity_cnt -self.hit_activity_check_times )
        print("Miss activity list:")

        return miss_activity_ratio


    
    def check_activity_prediction_situation(self):

        # todo: check the hit accout, and the hit duration
        
        #self.activity_begin_dict
        #self.activity_begin_list
        #self.activity_end_dict
        #self.activity_end_list


        #self.res_random_event_dict

        #self.res_hit_plus_miss_event_dict

        event_dict = self.res_hit_event_dict


        # duration
        sd = sorted(event_dict.items())
        pre_action = "empty"
        for k, v in sd:
            if v != pre_action:
                pre_action = v
            else:
                # update duration
                pass
            

        return ""


    def sensors_energy_time_costbk(self, action):
            """
            Calculate the energy and time cost
            :param action:
            :return:
            """

            energy_consum = 0
            time_cost = 0
            action_str = ACTION_DICT[action]


            # print("energy_time_cost action:", action)
            # # print(ACTION_DICT)
            # print("energy_time_cost action_str:", action_str)

            # interval = ACTION_INTERVAL_DICT[action]

            if 'audio' in action_str:
                energy_consum = energy_consum + (ENERGY_RECORDING_MIC - ENERGY_STANDBY) * AUDIO_RECORDING_TIME_COST   # 30
                energy_consum = energy_consum + AUDIO_FILE_SIZE / WIFI_BANDWIDTH_SPEED * (ENERGY_TX - ENERGY_STANDBY) # 50
                time_cost = time_cost + AUDIO_FILE_SIZE / WIFI_BANDWIDTH_SPEED

            if 'vision' in action_str:
                energy_consum = energy_consum + 0  #  ignore the image taking energy as it need less energy
                energy_consum = energy_consum + IMAGE_SIZE * IMAGE_COUNT / WIFI_BANDWIDTH_SPEED * (ENERGY_TX - ENERGY_STANDBY)
                time_cost = time_cost + IMAGE_SIZE * IMAGE_COUNT / WIFI_BANDWIDTH_SPEED

                # todo for tiny ML, without the transmission cost

            energy_consum = ACCELEROMETER_DATA_SIZE / WIFI_BANDWIDTH_SPEED * (ENERGY_TX - ENERGY_STANDBY)
            time_cost = time_cost + ACCELEROMETER_DATA_SIZE / WIFI_BANDWIDTH_SPEED

            time_cost = time_cost # do not need interval, just use sensors active time

            energy_consum = energy_consum # Do not need energy standby

            return energy_consum, time_cost

    def sensors_energy_time_cost(self, action):
            """
            Calculate the energy and time cost
            :param action:
            :return:
            """

            energy_consum = 0
            time_cost = 0
            action_str = ACTION_DICT[action]


            # print("energy_time_cost action:", action)
            # # print(ACTION_DICT)
            # print("energy_time_cost action_str:", action_str)

            # interval = ACTION_INTERVAL_DICT[action]

            if 'audio' in action_str:
                energy_consum = energy_consum + (ENERGY_RECORDING_MIC - ENERGY_STANDBY) * AUDIO_RECORDING_TIME_COST   # 30
                energy_consum = energy_consum + AUDIO_FILE_SIZE / WIFI_BANDWIDTH_SPEED * (ENERGY_TX - ENERGY_STANDBY) # 50
                # time_cost = time_cost + AUDIO_FILE_SIZE / WIFI_BANDWIDTH_SPEED

            if 'vision' in action_str:
                energy_consum = energy_consum + 0  #  ignore the image taking energy as it need less energy
                energy_consum = energy_consum + IMAGE_SIZE * IMAGE_COUNT / WIFI_BANDWIDTH_SPEED * (ENERGY_TX - ENERGY_STANDBY)
                # time_cost = time_cost + IMAGE_SIZE * IMAGE_COUNT / WIFI_BANDWIDTH_SPEED

                # todo for tiny ML, without the transmission cost

            energy_consum = ACCELEROMETER_DATA_SIZE / WIFI_BANDWIDTH_SPEED * (ENERGY_TX - ENERGY_STANDBY)
            # time_cost = time_cost + ACCELEROMETER_DATA_SIZE / WIFI_BANDWIDTH_SPEED

            time_cost = time_cost # do not need interval, just use sensors active time

            energy_consum = energy_consum # Do not need energy standby

            return energy_consum, time_cost

    def energy_time_cost(self, action):
        """
        Calculate the energy and time cost
        :param action:
        :return:
        """

        energy_consum = 0
        time_cost = 0
        action_str = ACTION_DICT[action]


        # print("energy_time_cost action:", action)
        # # print(ACTION_DICT)
        # print("energy_time_cost action_str:", action_str)

        # interval = ACTION_INTERVAL_DICT[action]
        interval = self.motion_triggered_interval

        if 'audio' in action_str:
            energy_consum = energy_consum + (ENERGY_RECORDING_MIC - ENERGY_STANDBY) * AUDIO_RECORDING_TIME_COST
            energy_consum = energy_consum + AUDIO_FILE_SIZE / WIFI_BANDWIDTH_SPEED * (ENERGY_TX - ENERGY_STANDBY)
            time_cost = time_cost + AUDIO_RECORDING_TIME_COST + AUDIO_FILE_SIZE / WIFI_BANDWIDTH_SPEED

        if 'vision' in action_str:
            energy_consum = energy_consum + 0  #  ignore the image taking energy as it need less energy
            energy_consum = energy_consum + IMAGE_SIZE * IMAGE_COUNT / WIFI_BANDWIDTH_SPEED * (ENERGY_TX - ENERGY_STANDBY)
            time_cost = time_cost + IMAGE_SIZE * IMAGE_COUNT / WIFI_BANDWIDTH_SPEED + IMAGE_TAKING_TIME_COST * IMAGE_COUNT

        # if 'accelerometer' in action_str:
        #     energy_consum = energy_consum + 0 + ACCELEROMETER_DATA_SIZE / WIFI_BANDWIDTH_SPEED * (ENERGY_TX - ENERGY_STANDBY)
        #     time_cost = time_cost + ACCELEROMETER_DATA_RECORDING_TIME_COST + ACCELEROMETER_DATA_SIZE / WIFI_BANDWIDTH_SPEED

        time_cost = time_cost + interval
        # activity, activity_priority, beginning_activity, end_activity = self.get_activity_by_action(action)
        #  if activity == '', in this case as we do not get the accurate activity from dataset, we just ignore this
        # if activity == -1:
        #     energy_consum = 0

        energy_consum = energy_consum + time_cost * ENERGY_STANDBY

        return energy_consum, time_cost

    def get_current_hour_time_real(self):
        now = datetime.now()
        dt_string = now.strftime(DATE_HOUR_TIME_FORMAT)
        # cur_hour_time = dt_string.strftime(DATE_HOUR_TIME_FORMAT).split()[1]
        # cur = datetime.strptime(cur_hour_time, HOUR_TIME_FORMAT)
        return dt_string
    
    def get_current_hour_time(self):
        cur_hour_time = self.running_time.strftime(DATE_HOUR_TIME_FORMAT).split()[1]
        cur = datetime.strptime(cur_hour_time, HOUR_TIME_FORMAT)
        return cur

    def get_current_date_day_time(self):
        cur_day_time = self.running_time.strftime(DATE_HOUR_TIME_FORMAT).split()[0]
        cur = datetime.strptime(cur_day_time, DAY_FORMAT_STR)
        return cur

    def get_train_running_time(self):

        final_time = self.running_time
        # format_time = str(final_time.hour) + str(final_time.minute) \
        #      + str(final_time.second)

        day_time_running = datetime.strptime("08:00:00", HOUR_TIME_FORMAT)
        target_day_time = day_time_running.replace(hour = final_time.hour, minute = final_time.minute, second = final_time.second)

        return target_day_time.timestamp()

    def get_current_running_time(self, time_cost):
        # Add time_cost (seconds) to datetime object
        final_time = self.running_time + timedelta(seconds = time_cost)
        if DEBUG:
            print('Get_current_running_time: ', final_time)



        return final_time

    def get_activity_by_action(self, action):
        activity = 0
        priority = 0

        accuracy = 0

        # wait for interval seconds and then trigger the sensors to collect data
        # interval = ACTION_INTERVAL_DICT[action]
        interval = self.motion_triggered_interval
        sensor_run_time = self.get_current_running_time(interval).strftime(DATE_HOUR_TIME_FORMAT)

        activity_str, beginning_activity, end_activity = tools_ascc.get_activity(self.activity_date_dict, self.activity_begin_list,
                                                                      self.activity_end_list, sensor_run_time)
        activity = tools_ascc.get_key(ACTIVITY_DICT, activity_str)

        priority = 3

        if activity == -1 or activity == -2:
            if DEBUG:
                print("get_activity_by_action-activity:", activity, "running time:", self.running_time, "activity_str:", activity_str)
            
            pass 
        else:
            # get activity priority
            priority = ACTIVITY_PRIORITY_DICT[activity]


        return activity, priority, beginning_activity, end_activity



    def trigger_acceleration(self):
        # todo, define different activity threshold, also, they should be related to the daily activities, daily time
        # RL use this data to predict next activity
        return 2.2

    # TODO: Implement the cnn model for motion-based activitiy recognition,  short walk, long walk, sitting, standing
    def check_motion_action_by_cnn(self, action, interval):
        # base_date = '2009-10-16'

        base_date = self.running_day

        motion_activity_dict = self.get_motion_activity(base_date)
        #print(motion_activity_dict)
        #motion_activity_dict = sorted(motion_activity_dict.items())

        

        running_time = self.running_time

        time_cost = ACTION_INTERVAL_DICT[action]

        motion_trigger_action = MOTION_TRIGGERRED_ACTION
        sensors_power_consumption, sensors_time_cost = self.sensors_energy_time_cost(action)

        acc_ratio = 0.80
       



    
        # '2009-10-16 08:42:01': 1, '2009-10-16 08:43:59': 1,
        for i in range(interval):
            new_time = running_time + timedelta(seconds = i)
            
            for key in motion_activity_dict.keys():
                motion_time_t = datetime.strptime(key, DATE_HOUR_TIME_FORMAT)

                # # 0, 1, 2, 3
                for tmp_t in range(4):
                    motion_time = motion_time_t + timedelta(seconds = tmp_t)

                # todo: motion time: range [-3, 3]
                # motion_time = datetime.strptime(key, DATE_HOUR_TIME_FORMAT) + 3
                # motion_time = datetime.strptime(key, DATE_HOUR_TIME_FORMAT) -3 

                # print((motion_time - new_time).seconds)
                # print("motion_time:", motion_time)
                # print("new_time:", new_time)

                
                # final_time = running_time
                # day_time_running = datetime.strptime("08:00:00", HOUR_TIME_FORMAT)
                # target_motion_time = day_time_running.replace(hour = final_time.hour, minute = final_time.minute, second = final_time.second)
                # print(target_motion_time.timestamp())
                # exit(0)

                    if (motion_time.timestamp() - new_time.timestamp()) <= 0 \
                        and (motion_time.timestamp() - running_time.timestamp()) >= 0:
                        
                        print("running time", self.running_time)
                        print("motion_time:", motion_time)
                        print("new_time:", new_time)
                        print("motion_time > new_time, motion triggerred")
                        time_cost = i
                        print("time cost:", time_cost)

                        # random_t = random.random()
                        # print('random_t:', random_t)
                        # if random_t > acc_ratio:
                        #     print('transition motion acc ratio occurs:', random_t)
                        #     self.running_time = self.get_current_running_time(time_cost)

                        #     continue

                        if not self.done:
                            self.motion_triggered_times = self.motion_triggered_times + 1

                        return True, time_cost
                

        return False, time_cost


    def check_motion_action(self, action, interval):
        # base_date = '2009-10-16'

        base_date = self.running_day

        motion_activity_dict = self.get_motion_activity(base_date)
        #print(motion_activity_dict)
        #motion_activity_dict = sorted(motion_activity_dict.items())

        

        running_time = self.running_time

        time_cost = ACTION_INTERVAL_DICT[action]

        motion_trigger_action = MOTION_TRIGGERRED_ACTION
        sensors_power_consumption, sensors_time_cost = self.sensors_energy_time_cost(action)

        acc_ratio = 0.80
       



    
        # '2009-10-16 08:42:01': 1, '2009-10-16 08:43:59': 1,
        for i in range(interval):
            new_time = running_time + timedelta(seconds = i)
            
            for key in motion_activity_dict.keys():
                motion_time_t = datetime.strptime(key, DATE_HOUR_TIME_FORMAT)

                # # 0, 1, 2, 3
                for tmp_t in range(4):
                    motion_time = motion_time_t + timedelta(seconds = tmp_t)

                # todo: motion time: range [-3, 3]
                # motion_time = datetime.strptime(key, DATE_HOUR_TIME_FORMAT) + 3
                # motion_time = datetime.strptime(key, DATE_HOUR_TIME_FORMAT) -3 

                # print((motion_time - new_time).seconds)
                # print("motion_time:", motion_time)
                # print("new_time:", new_time)

                
                # final_time = running_time
                # day_time_running = datetime.strptime("08:00:00", HOUR_TIME_FORMAT)
                # target_motion_time = day_time_running.replace(hour = final_time.hour, minute = final_time.minute, second = final_time.second)
                # print(target_motion_time.timestamp())
                # exit(0)

                    if (motion_time.timestamp() - new_time.timestamp()) <= 0 \
                        and (motion_time.timestamp() - running_time.timestamp()) >= 0:
                        
                        print("running time", self.running_time)
                        print("motion_time:", motion_time)
                        print("new_time:", new_time)
                        print("motion_time > new_time, motion triggerred")
                        time_cost = i
                        print("time cost:", time_cost)

                        # random_t = random.random()
                        # print('random_t:', random_t)
                        # if random_t > acc_ratio:
                        #     print('transition motion acc ratio occurs:', random_t)
                        #     self.running_time = self.get_current_running_time(time_cost)

                        #     continue

                        if not self.done:
                            self.motion_triggered_times = self.motion_triggered_times + 1

                        return True, time_cost
                

        return False, time_cost


    def get_motion_activity(self, base_date):
        # base_date = '2009-10-16'
 
        motion_activity_dict = {}
        
        # for i in range(82):
        # day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR)
        day_time_train = base_date
        day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
            tools_ascc.get_activity_date(day_time_str)



        motion_activity_cnt = 0
        motion_activity_dict = {}

        # for a_i in range(len(activity_begin_dict)):
        for key in activity_begin_dict.keys():
            # print(key)
            # print(activity_begin_dict[key])
            # print(activity_end_dict[key])
            time_list_begin = activity_begin_dict[key]
            time_list_end = activity_end_dict[key]
            for t_i in range(len(time_list_begin)):
                a_begin = datetime.strptime(time_list_begin[t_i], DATE_HOUR_TIME_FORMAT)
                try:
                    a_end = datetime.strptime(time_list_end[t_i], DATE_HOUR_TIME_FORMAT)
                except:
                    # print("End list not good", len(time_list_begin), len(time_list_end))
                    break
                # print(a_begin)
                # print(a_end)

                motion_activity_cnt = motion_activity_cnt + 1

                a_bein_str = a_begin.strftime(DATE_HOUR_TIME_FORMAT)

                motion_activity_dict[a_bein_str] = 1



        # print("motion_activity_cnt:", motion_activity_cnt)


        # begin, end = get_day_begin_end(activity_date_dict, activity_begin_dict, activity_end_dict)
        # print("Date:", day_time_str, " Begin:", begin.strftime(HOUR_TIME_FORMAT), " End:", end.strftime(HOUR_TIME_FORMAT),
        #       " Duration(Hours):", (end-begin).seconds/3600.0)
        
        return motion_activity_dict

    def get_running_time(self):
        return self.running_time

    def get_battery_level(self):
        wmu_cam_times = self.wmu_cam_times
        battery_level = 0
        if wmu_cam_times < 300:
            battery_level = 0
        elif wmu_cam_times < 600:
            battery_level = 1
        elif wmu_cam_times < 1200:
            battery_level = 2
        else:
            battery_level = 3
        return battery_level



    def set_current_running_time(self, time_cost):
        # Add time_cost (seconds) to datetime object
        final_time = self.running_time + timedelta(seconds = time_cost)
        if DEBUG:
            print('Get_current_running_time: ', final_time)

        self.running_time = final_time


        if self.get_current_date_day_time() > self.running_day:
            print('Get current date_day_time:', self.get_current_date_day_time(), " running day:", self.running_day)
            done = True
            self.done = True
