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
import imp
import random
import time
from typing import Dict
import numpy as np
# import tkinter as tk
from PIL import Image

from datetime import datetime
from datetime import timedelta

import log
import logging



import tools_ascc
import adl_type_constants

import env_socket_type_constants
import adl_env_client_lib


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


np.random.seed(1)

UNIT = 100
HEIGHT = 5
WIDTH = 5

ENERGY_STANDBY = 124  # mA
ENERGY_TX = 220  # mA
ENERGY_RECORDING_MIC = 130  # mA (to be Done from Ricky)


IMAGE_SIZE = 24.86  # KB
IMAGE_COUNT = 10  # 10 Frames
IMAGE_TAKING_TIME_COST = 1/30  # 30 frames

# AUDIO_FILE_SIZE = 50  # KB, Duration: 5 seconds
# AUDIO_RECORDING_TIME_COST = 5  # 5 seconds


AUDIO_FILE_SIZE = 87  # KB, Duration: 1 seconds
AUDIO_RECORDING_TIME_COST = 1  # 1 seconds



ACCELEROMETER_DATA_SIZE = 1  # KB  float 4 bytes,  3-axis,  data rate 10 HZ, recording time 5 seconds, size= 4 * 3 * 10 * 5 = 600 bytes
ACCELEROMETER_DATA_RECORDING_TIME_COST = 5  # 5 seconds (Reference: xxx)

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

AUDIO_ACTION = 0
VISION_ACTION = 1
MOTION_ACTION = 2
FUSION_ACTION = 3

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

ACTION_DICT = {
    0: "audio with 1 interval",  # "audio",
    1: "vision with 1/3 interval", #"vision",
    2: "motion with 2 interval", #"motion"
    3: "audio vision (fusion) with 2+1/3 interval",  # "fusion",


}


# todo: time cost for recognition:  image: 1.15 for 10 images, audio:0.1, motion: 1.14, totally: 2.5
ACTION_INTERVAL_DICT = {
    0: 1,  # "audio",
    1: 1.0/3, #"vision",
    2: 3, #"motion"
    3: 4+1.0/3, # fusion
}

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



class EnvASCC():
    def __init__(self, time_str = '2009-12-11 08:00:00'):

        self.running_time = datetime.now()
        self.day_begin = datetime.now()

        self.total_check_times = 0
        self.fusion_check_times = 0
        self.motion_check_times = 0

        self.sensor_time_cost = 0
        self.sensor_energy_cost = 0

        log.init_log("./log/env")  # ./log/my_program.log./log/my_program.log.wf7
        logging.info("Hello World!!!")


    def display_info(self):

        print("Date:", self.running_day, " Begin:", self.day_begin.strftime(HOUR_TIME_FORMAT), 
        " End:", self.day_end.strftime(HOUR_TIME_FORMAT),
              " Duration(Hours):", (self.day_end - self.day_begin).seconds/3600.0)
        print("Initial Battery Life:", BATTERY_LIFE)


    def step(self, p_action):

        logging.info("Send cmd to WMU to take fusion data, action %s ", p_action)

        try:
            if p_action == FUSION_ACTION:
                adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_IPRECEIVE, adl_type_constants.WMU_RECEIVE_PORT,
                                                            adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_FUSION)
                self.fusion_check_times += 1

            elif p_action == MOTION_ACTION:
                adl_env_client_lib.cmd_mode_sending_handler(adl_type_constants.WMU_IPRECEIVE, adl_type_constants.WMU_RECEIVE_PORT,
                                                            adl_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_MOTION)
                self.motion_check_times += 1
        
        except Exception as e:
            print("Got error when Send cmd to WMU, err:", e)
            logging.warn('Got error when Send cmd to WMU')
            logging.warn(e)

        
        self.total_check_times = self.total_check_times + 1



        action = p_action
        sensors_power_consumption, sensors_time_cost = self.sensors_energy_time_cost(action)

        if DEBUG:
            print("Step--p_action power_consumption:", sensors_power_consumption, " time cost:", sensors_time_cost)

        self.sensor_time_cost = self.sensor_time_cost + sensors_time_cost 
        self.sensor_energy_cost = self.sensor_energy_cost + sensors_power_consumption

        self.running_time = self.get_current_running_time()


        return 0

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
                time_cost = time_cost + AUDIO_RECORDING_TIME_COST + AUDIO_FILE_SIZE / WIFI_BANDWIDTH_SPEED

            if 'vision' in action_str:
                energy_consum = energy_consum + 0  #  ignore the image taking energy as it need less energy
                energy_consum = energy_consum + IMAGE_SIZE * IMAGE_COUNT / WIFI_BANDWIDTH_SPEED * (ENERGY_TX - ENERGY_STANDBY)
                time_cost = time_cost + IMAGE_SIZE * IMAGE_COUNT / WIFI_BANDWIDTH_SPEED + IMAGE_TAKING_TIME_COST * IMAGE_COUNT

            time_cost = time_cost # do not need interval, just use sensors active time

            energy_consum = energy_consum # Do not need energy standby

            return energy_consum, time_cost

    # def get_current_hour_time(self):
    #     cur_hour_time = self.running_time.strftime(DATE_HOUR_TIME_FORMAT).split()[1]
    #     cur = datetime.strptime(cur_hour_time, HOUR_TIME_FORMAT)
    #     return cur

    # def get_current_date_day_time(self):
    #     cur_day_time = self.running_time.strftime(DATE_HOUR_TIME_FORMAT).split()[0]
    #     cur = datetime.strptime(cur_day_time, DAY_FORMAT_STR)
    #     return cur


    def get_current_running_time(self):
        final_time = datetime.today()

        if DEBUG:
            print('Get_current_running_time: ', final_time)

        return final_time

    def get_running_time(self):
        return self.running_time

