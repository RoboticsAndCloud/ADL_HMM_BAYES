"""
@Brief: This file provides basic functions for converting the online daily activity data to
         the relevant dict. 
@Author: Fei.Liang
@Date: 08/12/2021

"""

"""
activity_date_dict = {
    "2009-10-16 00:01:04.000059": "Sleep",
    "2009-10-16 00:01:04.000059": "",
}
"""

"""
day_time_str: 2009-10-16
130.86667
day_time_str: 2009-10-17
90.71667
day_time_str: 2009-10-18
154.3
day_time_str: 2009-10-19
107.25
day_time_str: 2009-10-20
83.816666
day_time_str: 2009-10-21
115.78333
day_time_str: 2009-10-22
48.45
day_time_str: 2009-10-23
100.4
day_time_str: 2009-10-24
268.75
day_time_str: 2009-10-25
116.65
day_time_str: 2009-10-26
67.25
day_time_str: 2009-11-06
12.233334
day_time_str: 2009-11-18
64.95
day_time_str: 2009-11-19
End list not good 4 3
113.28333
day_time_str: 2009-11-20
116.51667
day_time_str: 2009-11-21
309.55
day_time_str: 2009-11-22
89.88333
day_time_str: 2009-11-23
End list not good 5 4
153.91667
day_time_str: 2009-11-24
123.0
day_time_str: 2009-11-25
End list not good 2 1
123.23333
day_time_str: 2009-11-26
464.43332
day_time_str: 2009-11-27
178.56667
day_time_str: 2009-11-28
73.73333
day_time_str: 2009-11-29
121.2
day_time_str: 2009-11-30
139.98334
day_time_str: 2009-12-01
125.25
day_time_str: 2009-12-02
End list not good 3 2
55.8
day_time_str: 2009-12-03
113.0
day_time_str: 2009-12-04
78.13333
day_time_str: 2009-12-05
77.433334
day_time_str: 2009-12-06
97.2
day_time_str: 2009-12-07
177.13333
day_time_str: 2009-12-08
108.25
day_time_str: 2009-12-09
107.03333
day_time_str: 2009-12-10
124.333336
day_time_str: 2009-12-11
52.016666
day_time_str: 2009-12-12
93.85
day_time_str: 2009-12-13
149.71666
day_time_str: 2009-12-14
120.11667
day_time_str: 2009-12-15
125.2
day_time_str: 2009-12-16
95.15
day_time_str: 2009-12-17
105.88333
day_time_str: 2009-12-18
100.333336
day_time_str: 2009-12-19
117.933334
day_time_str: 2009-12-20
154.21666
day_time_str: 2009-12-21
122.98333
day_time_str: 2009-12-22
217.81667
day_time_str: 2009-12-23
122.53333
day_time_str: 2009-12-24
88.25
day_time_str: 2009-12-25
End list not good 3 2
104.4
day_time_str: 2009-12-26
200.08333
day_time_str: 2009-12-27
78.916664
day_time_str: 2009-12-28
172.91667
day_time_str: 2009-12-29
132.23334
day_time_str: 2009-12-30
96.26667
day_time_str: 2009-12-31
End list not good 4 3
147.13333
day_time_str: 2010-01-01
94.48333
day_time_str: 2010-01-02
64.05
day_time_str: 2010-01-03
62.3
day_time_str: 2010-01-04
132.96666
day_time_str: 2010-01-05
56.166668
"""


from datetime import datetime
from datetime import timedelta
from genericpath import exists
import os
import re
import time

import constants

import numpy as np


from matplotlib import image

DEBUG = True

ASCC_DATA_NOTICE_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/ascc_data/notice.txt'
ASCC_DATA_RES_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/ascc_data/recognition_result.txt'
ASCC_DATA_SET_DIR = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/'
ASCC_DATASET_DATE_HOUR_TIME_FORMAT_DIR = '%Y-%m-%d-%H-%M-%S'

ASCC_AUDIO_DATA_NOTICE_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/ascc_data/notice.txt'
ASCC_AUDIO_DATA_RES_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/ascc_data/recognition_result.txt'

MILAN_BASE_DATE = '2009-10-16'
DATASET_TRAIN_DAYS = 82 # includes the days without acitivity data, totally 82 days


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
    14: "Master_Bedroom_Activity"
    # 15: "Phone_Call",
    # 16: "Fall",
    # 17: "Medication"
}


ACTIVITY_LIST = ["Desk_Activity",
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

"""
    From activity recognition part
    class_names=['bathroom','bedroom', 'morning_med', 'reading', 'kitchen','livingroom', 'chores', 'desk_activity', 'dining_room_activity',
                 'eve_med', 'leaving_home', 'meditate']
"""

ACTIVITY_MAPPING = {
    'bathroom': ['Guest_Bathroom', 'Master_Bathroom'],
    'bedroom' : ['Master_Bedroom_Activity'],
    'morning_med': ['Morning_Meds'],
    'reading': ['Read'],
    'kitchen': ['Kitchen_Activity'],
    'livingroom': ['Watch_TV'],
    'chores': ['Chores'],
    'desk_activity': ['Desk_Activity'],
    'dining_room_activity': ['Dining_Rm_Activity'],
    'eve_med': ['Eve_Meds'],
    'leaving_home': ['Leave_Home'],
    'meditate': ['Meditate']
}


ACTIVITY_LOCATION_MAPPING = {
    'bathroom': constants.LOCATION_BATHROOM,
    'bedroom' : constants.LOCATION_BEDROOM,
    'morning_med': constants.LOCATION_KITCHEN,
    'reading': constants.LOCATION_READINGROOM,
    'kitchen': constants.ACTIVITY_KITCHEN,
    'livingroom': constants.LOCATION_LIVINGROOM,
    'chores': ['Chores'],
    'desk_activity': constants.LOCATION_LIVINGROOM,
    'dining_room_activity': constants.LOCATION_DININGROOM,
    'eve_med': constants.ACTIVITY_KITCHEN,
    'leaving_home': constants.LOCATION_DOOR,
    'meditate': constants.LOCATION_BEDROOM
}


# labels = ['door_open_closed', 'eating', 'keyboard', 'pouring_water_into_glass', 'toothbrushing', 'vacuum',
#  'drinking', 'flush_toilet', 'microwave', 'quiet', 'tv_news', 'washing_hand']

ACTIVITY_AUDIO_MAPPING = {
    'toothbrushing': ['Guest_Bathroom', 'Master_Bathroom'],
    'flush_toilet': ['Guest_Bathroom', 'Master_Bathroom'],
    'washing_hand': ['Guest_Bathroom', 'Master_Bathroom'],

    'quiet' : ['Master_Bedroom_Activity'],

    'eating': ['Kitchen_Activity'],
    'pouring_water_into_glass': ['Kitchen_Activity'],
    'drinking': ['Kitchen_Activity'],
    'microwave': ['Kitchen_Activity'],

    'tv_news': ['Watch_TV'],
    'vacuum': ['Chores'],
    'keyboard': ['Desk_Activity'],
    'door_open_closed': ['Leave_Home'],
}

BEGIN_END_ACTIVITY_DURATION = 60 * 2  # if it happens in the first/last 11 seconds, it should be beginning/end activity

"""
2009-10-16 15:52:25.000010
Scenario:

Kitchen_Activity begin

Leave_Home begin
Leave_Home end

Kitchen_Activity end

"""

DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
HOUR_TIME_FORMAT = "%H:%M:%S"
DAY_FORMAT_STR = '%Y-%m-%d'


# Seconds, if the time does not match the records, try to get the activity around
ACTIVITY_TOLERANCE = 60 * 2 

ACTIVITY_NOON_HOUR = 12
ACTIVITY_NIGHT_HOUR = 18


activity_set_for_lstm = set()

# the Activity Node for HMM
class Activity_Node_Observable:
    def __init__(self, a_name, time_type, a_duration = 0):
        self.name = a_name
        self.time_type = time_type # Moring, Afternoon, Night
        self.duration = self.duration_converter(a_duration) # 5, 30, 60
    
    def get_info(self):
        return self.name, self.time_type
    
    def activity_res_generation(self):
        return self.name + '_' + str(self.time_type) + '_' + str(self.duration)
    
    def duration_converter(self, duration):
        res = 0
        if duration == 0:
            res = 0
        elif duration <= 5:
            res = 5
        elif duration <= 30:
            res = 30
        else:
            res = 60
        
        return res


class Dictlist(dict):
    def __setitem__(self, key, value):
        try:
            self[key]
        except KeyError:
            super(Dictlist, self).__setitem__(key, [])
        self[key].append(value)


def checkKey(dict, key):
    if key in dict:

        return True

    return False

def get_activity_date(data_str):

    # 2009-10-16 00:01:04.000059	M017	ON
    f = open("data", "r")
    # print(f.readline())

    activity_date_dict = {

    }

    activity_begin_dict = Dictlist()
    activity_end_dict = Dictlist()

    activity_begin_list = []
    activity_end_list = []

    activity = ""
    activity_str = ""

    stack = ["Sleep begin"] #  initially Sleep in the mid night

    for x in f:

        if data_str not in x:
            continue

        arr_tmp = x.split("\t")

        # 2009-10-16 03:55:50.000029	M021	OFF	Sleep end
        tmp_d = get_date_accurate(arr_tmp[0])
        target_d = get_date_day(data_str)

        if target_d < (tmp_d - timedelta(days = 1)):
            break

        activity = arr_tmp[-1].rstrip('\n').lstrip(' ')

        if 'begin' in activity:
            # print(activity)
            stack.append(activity)

            activity_str = activity.split()[0]
            activity_begin_dict[activity_str] = arr_tmp[0].split('.')[0]
            activity_begin_list.append(arr_tmp[0].split('.')[0])


        if len(stack) > 0:
            activity_str = stack[-1].split()[0]

        activity_date_time = arr_tmp[0].split('.')[0]

        if activity_str in ACTIVITY_LIST:
            activity_date_dict[activity_date_time] = activity_str
            #print(activity_str)

        if 'end' in activity:

            activity_str = activity.split()[0]
            activity_end_dict[activity_str] = arr_tmp[0].split('.')[0]
            activity_end_list.append(arr_tmp[0].split('.')[0])

            stack.pop()
            if activity_str == "" and len(stack) > 0:
                print("Activity_str should not be empty:", activity_str)
                print(x)

            if len(stack) > 0:
                activity_str = stack[-1]
            else:
                activity_str = ""

            # print(activity)

        if activity_str == "":
            if activity_date_dict.get(activity_date_time) is None:
                activity_date_dict[activity_date_time] = activity_str


    f.close()

    # print(activity_date_dict)
    # print(activity_begin_dict)
    # print(activity_end_dict)
    # print(activity_begin_list)
    # print(activity_end_list)
    return activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list

def get_date_day(time_str):
    # time_str = '2009-10-16' , 2009-10-16 03:55:50.000029	M021	OFF	Sleep end
    date_format_str = '%Y-%m-%d'
    format_str = time_str.split()[0]
    d = datetime.strptime(format_str, date_format_str)
    return d

def get_date_accurate(time_str):
    # time_str = '2009-10-16'
    #date_format_str = '%Y-%m-%d'
    format_str = time_str.split('.')[0]
    date_format_str = '%Y-%m-%d %H:%M:%S'
    d = datetime.strptime(format_str, date_format_str)
    return d


# function to return key for any value
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key

    if val == "":
        return -1

    return -2


def get_activity(activity_date_dict, activity_begin_list, activity_end_list, time_str):
    date_format_str = '%Y-%m-%d %H:%M:%S'
    if DEBUG:
        print("get_activity time_str:", time_str)
    d_act = datetime.strptime(time_str, date_format_str)
    begin_act = ""
    end_act = ""

    beginning_activity = False
    end_activity = False

    for begin_date_str in activity_begin_list:
        tmp_d = datetime.strptime(begin_date_str, date_format_str)

        if tmp_d <= d_act:
            begin_act = begin_date_str
            continue
        else:
            # check if it is begin activity
            if begin_act != "":
                d_begin_act = datetime.strptime(begin_act, date_format_str)
                
                d_act_test_begin = d_act - timedelta(seconds=BEGIN_END_ACTIVITY_DURATION)
                if d_act_test_begin < d_begin_act:
                    beginning_activity = True
            break

    for end_date_str in activity_end_list:
        tmp_d = datetime.strptime(end_date_str, date_format_str)

        if tmp_d < d_act:
            continue
        else:
            end_act = end_date_str
            # check if it is end activity
            if end_act != "":
                d_begin_act = datetime.strptime(end_act, date_format_str)
                d_act_test_begin = d_act + timedelta(seconds=BEGIN_END_ACTIVITY_DURATION)
                if d_act_test_begin > d_begin_act:
                    end_activity = True
            break

    str_d_act = d_act.strftime(DATE_TIME_FORMAT)
    
    # print(activity_date_dict)
    '''
    Given a time, it may not have the relavant activity record, so we check the 30 seconds instead
    KeyError: '2009-10-16 08:44:25'
    2009-10-16 08:44:17.000043	M026	OFF
    2009-10-16 08:44:24	M026	ON
    2009-10-16 08:44:26.000088	M008	ON

    '''

    if DEBUG:
        print("Get value from activity_date_dict== time:", str_d_act)

    activity_str = None
    counter_s = 0 # try to search the activity aroud the d_act
    while True:
        # if checkKey(activity_date_dict, str_d_act):
        #     activity_str = activity_date_dict[str_d_act]
        #     break

        # increse the d_act by 1 second
        d_act_add = d_act + timedelta(seconds = counter_s)
        str_d_act_add = d_act_add.strftime(DATE_TIME_FORMAT)
        # print("str_d_act_add:", str_d_act_add)
        
        if checkKey(activity_date_dict, str_d_act_add):
            activity_str = activity_date_dict[str_d_act_add]
            break

        # decrese the d_act by 1 second
        counter_s_o = -counter_s
        d_act_de = d_act + timedelta(seconds = counter_s_o)
        str_d_act_de = d_act_de.strftime(DATE_TIME_FORMAT)

        # print("str_d_act_de:", str_d_act_de)

        if checkKey(activity_date_dict, str_d_act_de):
            activity_str = activity_date_dict[str_d_act_de]
            break

        if counter_s > ACTIVITY_TOLERANCE:
            break

        counter_s = counter_s + 1

    if DEBUG:
        print("Get value from activity_date_dict== time:", str_d_act, "activity_str:", activity_str, "counter_s:", counter_s)



    return activity_str, beginning_activity, end_activity


def write_notice_into_file(file_name, data):
    with open(file_name, 'w') as f:
        f.write(str(data))
        f.close()
    
    return True

def read_res_from_file(file_name):
    with open(file_name, 'r') as f:
        res = str(f.read().strip())
        f.close()

    return res

# '2009-10-16 08:42:01' =>  2009-12-11-15-28-49
def convert_time_to_real(c_time):
    cur = c_time    
    ascc_activity_time = datetime.fromtimestamp(cur.timestamp())

    ascc_time_str = ascc_activity_time.strftime(ASCC_DATASET_DATE_HOUR_TIME_FORMAT_DIR)

    return ascc_time_str


def get_activity_by_audio_dnn(time_str, action='vision'):
    
    date_format_str = '%Y-%m-%d %H:%M:%S'
    if DEBUG:
        print("get_activity time_str:", time_str)
    d_act = datetime.strptime(time_str, date_format_str)

    image_dir_name = get_exist_image_dir(time_str, action)
    if image_dir_name == '':
        return ['None']
    print('===:', image_dir_name)

    audio_dir_name = image_dir_name.replace('Image', 'Audio')
    

    write_notice_into_file(ASCC_AUDIO_DATA_NOTICE_FILE, audio_dir_name)

    # wait for the result, 2-5 seconds, todo: test how long should wait, how to improve the speed
    time.sleep(1)

    # get the results
    res_str = read_res_from_file(ASCC_AUDIO_DATA_RES_FILE)

    
    print('Audio Recognition res:', res_str)

    res_list = res_str.split('\t')

    auido_type = res_str.split('(')[0]
    prob = res_str.split('(')[1].split(')')[0]

    res = auido_type


    # res = ACTIVITY_LOCATION_MAPPING[res_location]
    #print('res_activity_list:', res_activity_list)
    return res, prob


def get_exist_image_dir(time_str, action='vision'):
    if DEBUG:
        print("get_activity time_str:", time_str)
    d_act = datetime.strptime(time_str, DATE_TIME_FORMAT)

    for i in range( 60*60 ):
        new_time = d_act + timedelta(seconds = i)
        ascc_dir_time = convert_time_to_real(new_time)
        image_dir_name = ASCC_DATA_SET_DIR + '/' + 'Image/' + ascc_dir_time + '/'

        if os.path.exists(image_dir_name) == True:
            print('tools_ascc image_dir_name:', image_dir_name)

            return image_dir_name


    return ''

def sorter_take_count(elem):
    # print('elem:', elem)
    return elem[1]

def get_activity_by_vision_dnn(time_str, action='vision'):

    date_format_str = '%Y-%m-%d %H:%M:%S'
    if DEBUG:
        print("get_activity time_str:", time_str)
    d_act = datetime.strptime(time_str, date_format_str)

    image_dir_name = get_exist_image_dir(time_str, action)
    if image_dir_name == '':
        return ['None']
    print('===:', image_dir_name)
    

    write_notice_into_file(ASCC_DATA_NOTICE_FILE, image_dir_name)

    # wait for the result, 2-5 seconds, todo: test how long should wait, how to improve the speed
    time.sleep(5*2)

    # get the results
    res_str = read_res_from_file(ASCC_DATA_RES_FILE)

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
    print('res:', res_str)

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

    res = ACTIVITY_LOCATION_MAPPING[res_location]
    #print('res_activity_list:', res_activity_list)
    return res, max_location_prob



# Running DNN and get the result for activity recognition 
def get_activity_by_dnn(activity_date_dict, activity_begin_list, activity_end_list, time_str, action='vision'):
    date_format_str = '%Y-%m-%d %H:%M:%S'
    if DEBUG:
        print("get_activity time_str:", time_str)
    d_act = datetime.strptime(time_str, date_format_str)
    begin_act = ""
    end_act = ""

    beginning_activity = False
    end_activity = False



    ascc_dir_time = convert_time_to_real(d_act)
    
    vision_res_activity_list = []
    audio_res_activity_list = []

    if 'vision' in action:
        vision_res_activity_list = get_activity_by_vision_dnn(time_str, action)
        print('tools vision_res_activity_list:', vision_res_activity_list)

    
    if 'audio' in action:
        # audio_res_activity_list = get_activity_by_audio_dnn(time_str, action)
        audio_res_activity_list = get_activity_by_vision_dnn(time_str, action)

        print('tools audio_res_activity_list:', audio_res_activity_list)

        # vision_res_activity_list = get_activity_by_vision_dnn(time_str)
    
    if len(audio_res_activity_list) > 0 and len(vision_res_activity_list) > 0:
        audio_res = audio_res_activity_list[0]
        vision_res = vision_res_activity_list[0]
        if audio_res != vision_res:
            print('Diff in recognition for time:', time_str, 'audio_res: ', audio_res, ' vision_res:', vision_res)


    for begin_date_str in activity_begin_list:
        tmp_d = datetime.strptime(begin_date_str, date_format_str)

        if tmp_d <= d_act:
            begin_act = begin_date_str
            continue
        else:
            # check if it is begin activity
            if begin_act != "":
                d_begin_act = datetime.strptime(begin_act, date_format_str)
                
                d_act_test_begin = d_act - timedelta(seconds=BEGIN_END_ACTIVITY_DURATION)
                if d_act_test_begin < d_begin_act:
                    beginning_activity = True
            break

    for end_date_str in activity_end_list:
        tmp_d = datetime.strptime(end_date_str, date_format_str)

        if tmp_d < d_act:
            continue
        else:
            end_act = end_date_str
            # check if it is end activity
            if end_act != "":
                d_begin_act = datetime.strptime(end_act, date_format_str)
                d_act_test_begin = d_act + timedelta(seconds=BEGIN_END_ACTIVITY_DURATION)
                if d_act_test_begin > d_begin_act:
                    end_activity = True
            break

    str_d_act = d_act.strftime(DATE_TIME_FORMAT)
    
    # print(activity_date_dict)
    '''
    Given a time, it may not have the relavant activity record, so we check the 30 seconds instead
    KeyError: '2009-10-16 08:44:25'
    2009-10-16 08:44:17.000043	M026	OFF
    2009-10-16 08:44:24	M026	ON
    2009-10-16 08:44:26.000088	M008	ON

    '''

    if DEBUG:
        print("Get value from activity_date_dict== time:", str_d_act)

    # activity_str = None
    # counter_s = 0 # try to search the activity aroud the d_act
    # while True:
    #     # if checkKey(activity_date_dict, str_d_act):
    #     #     activity_str = activity_date_dict[str_d_act]
    #     #     break

    #     # increse the d_act by 1 second
    #     d_act_add = d_act + timedelta(seconds = counter_s)
    #     str_d_act_add = d_act_add.strftime(DATE_TIME_FORMAT)
    #     # print("str_d_act_add:", str_d_act_add)
        
    #     if checkKey(activity_date_dict, str_d_act_add):
    #         activity_str = activity_date_dict[str_d_act_add]
    #         break

    #     # decrese the d_act by 1 second
    #     counter_s_o = -counter_s
    #     d_act_de = d_act + timedelta(seconds = counter_s_o)
    #     str_d_act_de = d_act_de.strftime(DATE_TIME_FORMAT)

    #     # print("str_d_act_de:", str_d_act_de)

    #     if checkKey(activity_date_dict, str_d_act_de):
    #         activity_str = activity_date_dict[str_d_act_de]
    #         break

    #     if counter_s > ACTIVITY_TOLERANCE:
    #         break

    #     counter_s = counter_s + 1

    # todo audio res_activity_list

    # check the result: prefere to use vision result
    # 
    res_activity_list = []
    if len(audio_res_activity_list) > 0:
        res_activity_list = audio_res_activity_list

    if len(vision_res_activity_list) > 0:
        res_activity_list = vision_res_activity_list

    if DEBUG:
        print("Get value from activity_date_dict== time:", str_d_act, "res_activity_list:", res_activity_list, "counter_s:", counter_s)

    return res_activity_list, beginning_activity, end_activity

def check_activity_phase_begin(activity_date_dict, time_str):

    date_format_str = '%Y-%m-%d %H:%M:%S'
    d = datetime.strptime(time_str, date_format_str)
    d = d - timedelta(seconds=11)
    # print(d)
    # Convert datetime object to string in specific format
    test_time_str = d.strftime(date_format_str)

    if activity_date_dict[test_time_str] == activity_date_dict[time_str]:
        return True

    return False


def check_activity_phase_end(activity_date_dict, time_str):
    date_format_str = '%Y-%m-%d %H:%M:%S'
    d = datetime.strptime(time_str, date_format_str)
    d = d + timedelta(seconds=11)
    # print(d)
    # Convert datetime object to string in specific format
    test_time_str = d.strftime(date_format_str)

    if activity_date_dict[test_time_str] != activity_date_dict[time_str]:
        return True

    return False

def get_day_begin_end(activity_date_dict, activity_begin_dict, activity_end_dict):

    # activity_date_dict, activity_begin_list, activity_end_list= get_activity_date()

    sleep_activity = "Sleep"
    morning_meds_activity = "Morning_Meds"

    day_time_begin = datetime.strptime("08:00:00", HOUR_TIME_FORMAT)
    day_time_end = datetime.strptime("21:00:00", HOUR_TIME_FORMAT)

    morning_get_up_time_s = datetime.strptime("06:00:00", HOUR_TIME_FORMAT)
    morning_get_up_time_e = datetime.strptime("11:00:00", HOUR_TIME_FORMAT)

    night_sleep_time_s = datetime.strptime("19:00:00", HOUR_TIME_FORMAT)
    night_sleep_time_e = datetime.strptime("23:59:00", HOUR_TIME_FORMAT)

    '''
    begin dict
    {'Bed_to_Toilet': '2009-10-16 22:20:15', 'Sleep': '2009-10-16 22:21:23', 'Morning_Meds': '2009-10-16 08:42:01', 'Watch_TV': '2009-10-16 18:56:26', 'Kitchen_Activity': '2009-10-16 20:40:23', 'Chores': '2009-10-16 10:50:21', 'Leave_Home': '2009-10-16 18:50:58', 'Read': '2009-10-16 10:07:19', 'Guest_Bathroom': '2009-10-16 20:39:14', 'Master_Bathroom': '2009-10-16 20:41:12', 'Desk_Activity': '2009-10-16 18:52:14', 'Eve_Meds': '2009-10-16 20:36:20'}

    '''


    if checkKey(activity_end_dict, sleep_activity):
        sleep_time_end_list = activity_end_dict[sleep_activity]
        for i in range(len(sleep_time_end_list)):
            t = sleep_time_end_list[len(sleep_time_end_list) -i -1]
        # for t in sleep_time_end_list:
            sleep_time_end = datetime.strptime(t.split()[1], HOUR_TIME_FORMAT)
            if sleep_time_end > morning_get_up_time_s and sleep_time_end < morning_get_up_time_e:
                day_time_begin = sleep_time_end
                break
    elif checkKey(activity_begin_dict, morning_meds_activity):
        morning_meds_time_begin = \
            datetime.strptime(activity_begin_dict[morning_meds_activity][0].split()[1], HOUR_TIME_FORMAT)
        if morning_meds_time_begin > morning_get_up_time_s and morning_meds_time_begin < morning_get_up_time_e:
            day_time_begin = morning_meds_time_begin

    if checkKey(activity_begin_dict, sleep_activity):
        sleep_time_begin_list = activity_begin_dict[sleep_activity]
        for t in sleep_time_begin_list:
            sleep_time_begin = datetime.strptime(t.split()[1], HOUR_TIME_FORMAT)
            if sleep_time_begin > night_sleep_time_s and sleep_time_begin < night_sleep_time_e:
                day_time_end = sleep_time_begin
                break

    if checkKey(activity_end_dict, morning_meds_activity):
        morning_meds_time_end = \
            datetime.strptime(activity_end_dict[morning_meds_activity][0].split()[1], HOUR_TIME_FORMAT)

    # print("The day begin:", day_time_begin,  "The day end:", day_time_end)
    # print("morning_meds_time_begin_activity:", morning_meds_time_begin)

    return day_time_begin, day_time_end

def get_day_begin_end_summary():
    counter = 0
    for i in range(82):
        day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days = i)
        day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
            get_activity_date(day_time_str)

        if len(activity_begin_list) == 0:
            continue

        counter = counter + 1
        begin, end = get_day_begin_end(activity_date_dict, activity_begin_dict, activity_end_dict)
        print("Date:", day_time_str, " Begin:", begin.strftime(HOUR_TIME_FORMAT), " End:", end.strftime(HOUR_TIME_FORMAT),
              " Duration(Hours):", (end-begin).seconds/3600.0)

    print("Total days:", counter)

    return 0



def get_activity_count_state_list_by_date(base_date):
    counter = 0
    # base_date = '2009-12-11'


    day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days = 0)
   
    day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

    activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
        get_activity_date(day_time_str)

    print("=====================================")
    print("Date:", day_time_str)
    # print("activity_begin_dict", len(activity_begin_dict))
    # print("activity_end_dict", len(activity_end_dict))

    motion_activity_cnt = 0
    import collections
    output_dict = {}
    output_activity_dict = {}
    output_activity_symbol_dict = {}

    # print("Activity", "\t", "Start", "\t", "End", "\t", "Duration")

    activity_date_dict1 = {}
    activity_begin_dict1 = {}
    activity_end_dict1 = {}
    day_begin, day_end = get_day_begin_end(activity_date_dict, 
                                        activity_begin_dict, activity_end_dict)

    for key in activity_begin_dict.keys():
        time_list_begin = activity_begin_dict[key]
        time_list_end = activity_end_dict[key]
        for t_i in range(len(time_list_begin)):
            a_begin = datetime.strptime(time_list_begin[t_i], DATE_TIME_FORMAT)
            try:
                a_end = datetime.strptime(time_list_end[t_i], DATE_TIME_FORMAT)
            except:
                print("End list not good", len(time_list_begin), len(time_list_end))
                break

            tmp_a_begin = datetime.strptime(time_list_begin[t_i].split()[1], HOUR_TIME_FORMAT)
            if tmp_a_begin < day_begin:
                # print("A begin < day begin, ignore:", a_begin, day_begin)
                continue

            tmp_a_begin = datetime.strptime(time_list_begin[t_i].split()[1], HOUR_TIME_FORMAT)
            if tmp_a_begin >= day_end:
                # print("A begin > day end, ignore:", a_end, day_end)
                continue

            # tmp_a_end = datetime.strptime(time_list_end[t_i].split()[1], HOUR_TIME_FORMAT)
            # if tmp_a_end > day_end:
            #     print("A end > day end, ignore:", a_end, day_end)
            #     continue

            duration = (a_end - a_begin).seconds * 1.0 /60
            motion_activity_cnt = motion_activity_cnt + 1

            if duration > 0:
                tmp_str = key + "\t" + time_list_begin[t_i] + "\t" + time_list_end[t_i] + "\t" + str(duration)
                output_dict[time_list_begin[t_i]] = tmp_str
                # output_activity_dict[time_list_begin[t_i]] = key

                # print('t_begin:', time_list_begin[t_i], ':', tmp_a_begin)
                activity_type = 'M'
                if tmp_a_begin.hour < ACTIVITY_NOON_HOUR:
                    activity_type = 'M'
                elif tmp_a_begin.hour < ACTIVITY_NIGHT_HOUR:
                    activity_type = 'A'
                else:
                    activity_type = 'N'
                
                # print('activity_type:', activity_type)

                node = Activity_Node_Observable(key, activity_type, 0)

                output_activity_dict[time_list_begin[t_i]] = key

                output_activity_symbol_dict[time_list_begin[t_i]] = node.activity_res_generation()


                # todo:  key_type: check the begin time, and convert into Morning, Afternoon, Night
                
                # print(key, "\t", time_list_begin[t_i], "\t", time_list_end[t_i], "\t", duration)

    # if len(activity_begin_list) == 0:
    #     continue


    # sd = sorted(output_dict.items())
    # state_list = []

    # for k,v in sd:
    #     print(v)
    #     state_list.append(v)

    # print("motion_activity_cnt:", motion_activity_cnt)
    # print("len:", len(sd))

    sd = sorted(output_activity_dict.items())
    state_list = []
    for k,v in sd:
        # print(v)
        state_list.append(v)

    # print("motion_activity_cnt:", motion_activity_cnt)
    # print("len:", len(sd))
    state_list_tuple = tuple(state_list)


    sd_symbol = sorted(output_activity_symbol_dict.items())
    symbol_list = []
    for k,v in sd_symbol:
        # print(v)
        symbol_list.append(v)

    symbol_list_tuple = tuple(symbol_list)





    return motion_activity_cnt, state_list_tuple, symbol_list_tuple


def get_activity_count_by_date(base_date):
    counter = 0
    # base_date = '2009-12-11'


    day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days = 0)
   
    day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

    activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
        get_activity_date(day_time_str)

    print("=====================================")
    print("Date:", day_time_str)
    print("activity_begin_dict", len(activity_begin_dict))
    print("activity_end_dict", len(activity_end_dict))

    motion_activity_cnt = 0
    import collections
    output_dict = {}

    print("Activity", "\t", "Start", "\t", "End", "\t", "Duration")

    for key in activity_begin_dict.keys():
        time_list_begin = activity_begin_dict[key]
        time_list_end = activity_end_dict[key]
        for t_i in range(len(time_list_begin)):
            a_begin = datetime.strptime(time_list_begin[t_i], DATE_TIME_FORMAT)
            try:
                a_end = datetime.strptime(time_list_end[t_i], DATE_TIME_FORMAT)
            except:
                print("End list not good", len(time_list_begin), len(time_list_end))
                break

            duration = (a_end - a_begin).seconds * 1.0 /60
            motion_activity_cnt = motion_activity_cnt + 1

            if duration > 0:
                tmp_str = key + "\t" + time_list_begin[t_i] + "\t" + time_list_end[t_i] + "\t" + str(duration)
                output_dict[time_list_begin[t_i]] = tmp_str
                # print(key, "\t", time_list_begin[t_i], "\t", time_list_end[t_i], "\t", duration)

    # if len(activity_begin_list) == 0:
    #     continue

    state_list = []

    sd = sorted(output_dict.items())

    for k,v in sd:
        print(v)
        state_list.append(k)

    print("motion_activity_cnt:", motion_activity_cnt)
    print("len:", len(sd))


    return motion_activity_cnt

def get_activity_duration_by_date(base_date):
    counter = 0
    # base_date = '2009-12-11'


    day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days = 0)
   
    day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

    activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
        get_activity_date(day_time_str)

    # print("=====================================")
    # print("Date:", day_time_str)
    # print("activity_begin_dict", activity_begin_dict)
    # print("activity_end_dict", len(activity_end_dict))

    motion_activity_cnt = 0
    import collections
    output_dict = {}

    # print("Activity", "\t", "Start", "\t", "End", "\t", "Duration")

    activity_duration_list = []

    for key in activity_begin_dict.keys():
        time_list_begin = activity_begin_dict[key]
        time_list_end = activity_end_dict[key]
        for t_i in range(len(time_list_begin)):
            a_begin = datetime.strptime(time_list_begin[t_i], DATE_TIME_FORMAT)
            try:
                a_end = datetime.strptime(time_list_end[t_i], DATE_TIME_FORMAT)
            except:
                print("End list not good", len(time_list_begin), len(time_list_end))
                break

            # a_begin = a_begin.replace(year=2009, month=1, day=1)
            # a_end = a_end.replace(year=2009, month=1, day=1)
            # print(a_begin)
            # print(a_end)


            duration = (a_end - a_begin).seconds * 1.0 /60
            motion_activity_cnt = motion_activity_cnt + 1

            if key == 'Sleep':
                continue

            if duration > 0:
                tmp_str = key + "\t" + time_list_begin[t_i] + "\t" + time_list_end[t_i] + "\t" + str(duration)
                output_dict[time_list_begin[t_i]] = tmp_str
                # print(key, "\t", time_list_begin[t_i], "\t", time_list_end[t_i], "\t", duration)
                activity_index = key
                
                timestamp = time_list_begin[t_i]
                
                day_time = datetime.strptime(time_list_begin[t_i], DATE_TIME_FORMAT)

                day_time_running = datetime.strptime("08:00:00", HOUR_TIME_FORMAT)
                day_time = day_time_running.replace(hour = day_time.hour, minute = day_time.minute, second = day_time.second)

                timestamp = day_time.timestamp()
                activity_index = get_key(ACTIVITY_DICT, key)

                duration_list = [duration, timestamp, activity_index]
                activity_duration_list.append(duration_list)

                activity_set_for_lstm.add(activity_index)

    # if len(activity_begin_list) == 0:
    #     continue

    sd = sorted(output_dict.items())

    # for k,v in sd:
    #     print(v)

    # print("motion_activity_cnt:", motion_activity_cnt)
    # print("len:", len(sd))

    # print(activity_duration_list)
    temp = np.array(activity_duration_list, dtype=np.float32)

    # print(max(temp[:,0]))


    return activity_duration_list

def get_activity_for_state_list():

    res_activity_duration_list = []
    igore_list = []
    state_list_all = []
    symbol_list_all = []
    for i in range(DATASET_TRAIN_DAYS):
    # for i in range(2):

        base_date = MILAN_BASE_DATE
        day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days=i)
        day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
            get_activity_date(day_time_str)

        if len(activity_begin_list) == 0:
            continue

        if str(day_time_str) in igore_list:
            print("in ignore day_time_str:", day_time_str)
            continue
        
        # print("=============================================================================")
        print("day_time_str:", day_time_str)
        cnt, state_list, symbol_list = get_activity_count_state_list_by_date(day_time_str)
        # print('state_list, len:', len(state_list), ',state_list:',state_list, 'symbol_list:', symbol_list)
        if len(state_list) < 10:
            continue
        state_list_all.append(state_list)
        symbol_list_all.append(symbol_list)

        

        # exit(0)

    return state_list_all, symbol_list_all
    #return symbol_list_all, symbol_list_all
    #return state_list_all, state_list_all

def get_activity_information_all():

    res_activity_duration_list = []
    igore_list = []
    for i in range(DATASET_TRAIN_DAYS):
    # for i in range(2):

        base_date = MILAN_BASE_DATE
        day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days=i)
        day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
            get_activity_date(day_time_str)

        if len(activity_begin_list) == 0:
            continue

        if str(day_time_str) in igore_list:
            print("in ignore day_time_str:", day_time_str)
            continue
        
        print("=============================================================================")
        print("day_time_str:", day_time_str)
        get_activity_count_by_date(day_time_str)
        
        # res_activity_duration_list.append(activity_duration_list)

        # exit(0)

    return 0

def get_duration_from_dataset2():

    res_activity_duration_list = []
    igore_list = ['2009-12-24', '2009-12-25', '2009-12-31', '2010-01-01']
    for i in range(DATASET_TRAIN_DAYS):
    # for i in range(2):

        base_date = MILAN_BASE_DATE
        day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days=i)
        day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
            get_activity_date(day_time_str)

        if len(activity_begin_list) == 0:
            continue

        if str(day_time_str) in igore_list:
            print("in ignore day_time_str:", day_time_str)
            continue
        print("day_time_str:", day_time_str)

        activity_duration_list = get_activity_duration_by_date(day_time_str)
        res_activity_duration_list.append(activity_duration_list)

        # exit(0)

    return res_activity_duration_list


def get_duration_from_dataset():

    res_activity_duration_list = []
    igore_list = ['2009-12-24', '2009-12-25', '2009-12-31', '2010-01-01']
    for i in range(DATASET_TRAIN_DAYS):
    # for i in range(2):

        base_date = MILAN_BASE_DATE
        day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days=i)
        day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
            get_activity_date(day_time_str)

        if len(activity_begin_list) == 0:
            continue

        if str(day_time_str) in igore_list:
            print("in ignore day_time_str:", day_time_str)
            continue
        # print("day_time_str:", day_time_str)

        activity_duration_list = get_activity_duration_by_date(day_time_str)
        res_activity_duration_list.extend(activity_duration_list)

        # exit(0)
    print(activity_set_for_lstm)
    print('len activity set:',len(activity_set_for_lstm))

    return res_activity_duration_list

def get_activity_duration_output():
    counter = 0
    base_date = '2009-12-11'


    day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days = 0)
    day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

    activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
        get_activity_date(day_time_str)

    print("=====================================")
    print("Date:", day_time_str)
    print("activity_begin_dict", len(activity_begin_dict))
    print("activity_end_dict", len(activity_end_dict))

    motion_activity_cnt = 0
    import collections
    output_dict = {}

    print("Activity", "\t", "Start", "\t", "End", "\t", "Duration")

    for key in activity_begin_dict.keys():
        time_list_begin = activity_begin_dict[key]
        time_list_end = activity_end_dict[key]
        for t_i in range(len(time_list_begin)):
            a_begin = datetime.strptime(time_list_begin[t_i], DATE_TIME_FORMAT)
            try:
                a_end = datetime.strptime(time_list_end[t_i], DATE_TIME_FORMAT)
            except:
                print("End list not good", len(time_list_begin), len(time_list_end))
                break

            duration = (a_end - a_begin).seconds * 1.0 /60
            motion_activity_cnt = motion_activity_cnt + 1

            if duration > 0:
                tmp_str = key + "\t" + time_list_begin[t_i] + "\t" + time_list_end[t_i] + "\t" + str(duration)
                output_dict[time_list_begin[t_i]] = tmp_str
                # print(key, "\t", time_list_begin[t_i], "\t", time_list_end[t_i], "\t", duration)

    # if len(activity_begin_list) == 0:
    #     continue

    sd = sorted(output_dict.items())

    for k,v in sd:
        print(v)

    print("motion_activity_cnt:", motion_activity_cnt)
    print("len:", len(sd))


    return 0



def get_activity_duration():
    counter = 0
    base_date = '2009-10-16'
    # base_date = '2009-12-29'

    for i in range(82):
        day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days = i)
        day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
            get_activity_date(day_time_str)

        print("=====================================")
        print("Date:", day_time_str)
        print("activity_begin_dict", len(activity_begin_dict))
        print("activity_end_dict", len(activity_end_dict))

        motion_activity_cnt = 0

        # for a_i in range(len(activity_begin_dict)):
        for key in activity_begin_dict.keys():
            # print(key)
            # print(activity_begin_dict[key])
            # print(activity_end_dict[key])
            time_list_begin = activity_begin_dict[key]
            time_list_end = activity_end_dict[key]
            for t_i in range(len(time_list_begin)):
                a_begin = datetime.strptime(time_list_begin[t_i], DATE_TIME_FORMAT)
                try:
                    a_end = datetime.strptime(time_list_end[t_i], DATE_TIME_FORMAT)
                except:
                    print("End list not good", len(time_list_begin), len(time_list_end))
                    break
                # print(a_begin)
                # print(a_end)
                duration = (a_end - a_begin).seconds * 1.0 /60
                motion_activity_cnt = motion_activity_cnt + 1

                if duration > 0:
                    print(key, "\t", duration)

        print("motion_activity_cnt:", motion_activity_cnt)
        if len(activity_begin_list) == 0:
            continue

        counter = counter + 1
        # begin, end = get_day_begin_end(activity_date_dict, activity_begin_dict, activity_end_dict)
        # print("Date:", day_time_str, " Begin:", begin.strftime(HOUR_TIME_FORMAT), " End:", end.strftime(HOUR_TIME_FORMAT),
        #       " Duration(Hours):", (end-begin).seconds/3600.0)
        

    print("Total days:", counter)

    return 0


def get_motion_activity(base_date):
    counter = 0
    # base_date = '2009-10-16'
    # base_date = '2009-12-29'
    motion_activity_dict = {}
    
    for i in range(82):
        day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days = i)
        day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
            get_activity_date(day_time_str)



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
                a_begin = datetime.strptime(time_list_begin[t_i], DATE_TIME_FORMAT)
                try:
                    a_end = datetime.strptime(time_list_end[t_i], DATE_TIME_FORMAT)
                except:
                    # print("End list not good", len(time_list_begin), len(time_list_end))
                    break
                # print(a_begin)
                # print(a_end)
                duration = (a_end - a_begin).seconds * 1.0 /60
                motion_activity_cnt = motion_activity_cnt + 1

                a_bein_str = a_begin.strftime(DATE_TIME_FORMAT)

                motion_activity_dict[a_bein_str] = 1

                # if duration > 0:
                #     print(key, "\t", duration)

        print("motion_activity_cnt:", motion_activity_cnt)
        if len(activity_begin_list) == 0:
            continue

        counter = counter + 1
        # begin, end = get_day_begin_end(activity_date_dict, activity_begin_dict, activity_end_dict)
        # print("Date:", day_time_str, " Begin:", begin.strftime(HOUR_TIME_FORMAT), " End:", end.strftime(HOUR_TIME_FORMAT),
        #       " Duration(Hours):", (end-begin).seconds/3600.0)
        break
        

    print("Total days:", counter)

    return motion_activity_dict

def get_activity_by_day_str():
    day_time_train = datetime.strptime('2009-10-26', DAY_FORMAT_STR)
    day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

    activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
        get_activity_date(day_time_str)

    if len(activity_begin_list) == 0:
        print("activity_begin_list == 0")

    begin, end = get_day_begin_end(activity_date_dict, activity_begin_dict, activity_end_dict)
    print("Date:", day_time_str, " Begin:", begin.strftime(HOUR_TIME_FORMAT), " End:", end.strftime(HOUR_TIME_FORMAT),
            " Duration(Hours):", (end-begin).seconds/3600.0)

    activity_str, beginning_activity, end_activity = get_activity(activity_date_dict, activity_begin_list, activity_end_list, '2009-10-16 08:59:22' )
    print(activity_str, beginning_activity, end_activity)

    return 0



ACTION_INTERVAL_DICT = {
    0: 0,  # "audio",
    1: 0, #"vision",
    2: 0, #"audio_vision"

    3: 1,  # "audio",
    4: 1,  # "vision",
    5: 1,  # "audio_vision"

    6: 5,  # "audio",
    7: 5,  # "vision",
    8: 5,  # "audio_vision"

    9: 10,  # "audio",
    10: 10,  # "vision",
    11: 10,  # "audio_vision"

    12: 30,  # "audio",
    13: 30,  # "vision",
    14: 30,  # "audio_vision"

    15: 60,  # "audio",
    16: 60,  # "vision",
    17: 60,  # "audio_vision"

    18: 120,  # "audio",
    19: 120,  # "vision",
    20: 120,  # "audio_vision"
}

DATE_HOUR_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def check_motion_action(action, interval, time_str):
    base_date = '2009-10-16'
    motion_activity_dict = get_motion_activity(base_date)
    print(motion_activity_dict)

    # running time
    # interval
    # action = 15
    # interval = ACTION_INTERVAL_DICT[action]
    # time_str = '2009-10-16 08:41:50'
    running_time = datetime.strptime(time_str, DATE_HOUR_TIME_FORMAT)
    time_cost = 0

    motion_trigger_action = 3
   
    # '2009-10-16 08:42:01': 1, '2009-10-16 08:43:59': 1,
    for i in range(interval + 1):
        new_time = running_time + timedelta(seconds = i)
        
        for key in motion_activity_dict.keys():
            motion_time = datetime.strptime(key, DATE_HOUR_TIME_FORMAT)

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
                
                print("motion_time:", motion_time)
                print("new_time:", new_time)
                print("motion_time > new_time, motion triggerred")
                time_cost = i
                print("time cost:", time_cost)

                return motion_trigger_action
            

    return action

def get_activity_count_info():
    counts = dict()

    res = get_duration_from_dataset()
    total_cnt = 0
    for l in res:
        index = l[2]
        a = ACTIVITY_DICT[index]
        counts[a] = counts.get(a, 0) + 1
        total_cnt = total_cnt + 1

    p_counts = dict()
    for k in counts:
        p_counts[k] = counts[k] * 1.0 /total_cnt

    return counts, p_counts

def get_activity_duration_cnt_set():
    res = get_duration_from_dataset()

    act_dict = {}

    for lis in res:
        duration = lis[0]
        key = lis[2]
        if act_dict.get(key) == None:
            act_dict[key] = []
        
        act_dict[key].append(duration)

    res_dict = {}
    for key in act_dict.keys():
        tmp_list = act_dict[key]
        tmp_list.sort()
        act_str = ACTIVITY_DICT[key]
        res_dict[act_str] = tmp_list

    # print(res_dict)
    # print('============================================================================================================================')
    # print(res_dict[ACTIVITY_DICT[0]])

    return res_dict

if __name__ == "__main__":

    # get_activity_duration_prob()
    # res = get_duration_from_dataset()
    # print(res)
    # print(len(res))

    # ACTIVITY_DICT
    # lens = []
    # for i in res:
    #     print(len(i))
    #     lens.append(len(i))
    #     print(i)
    # print("max:", max(lens))

    

    # exit(0)

    # res, p_res = get_activity_count_info()
    # print('res:', res)
    # print()
    # print('p_res:', p_res)
    # exit(0)

    # get_activity_information_all()

    # state_list = get_activity_for_state_list()
    # print(state_list)
    # print(len(state_list))

    # for i in range(len(state_list)):
    #     print(state_list[i])
    #     print("==")


    # cnt = get_activity_count_by_date('2009-12-10')
    # print('cnt:', cnt)
    # res = get_activity_duration_by_date('2009-11-26')
    # print(res)

    # res = get_duration_from_dataset2()
    # print(res)
    # print(len(res))
    # lens = []
    # for i in res:
    #     print(len(i))
    #     lens.append(len(i))
    #     print(i)
    # print("max:", max(lens))

    
    # print(max(res[:, 0]))

    # exit(0)

    time_str = '2009-12-11 09:10:46' # leaving home door
    # time_str = '2009-12-11 10:06:39' # reading
    # time_str = '2009-12-11 12:24:07'
    time_str = '2009-12-11 08:45:27'
    # /home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309//Image/
    # 2009-12-11-22-47-06/
    time_str = '2009-12-11 22:47:06'
    time_str = '2009-12-11 16:42:33'
    # time_str = '2009-12-11 16:42:44'

    time_str = '2009-12-11 20:49:36'
    time_str = '2009-12-11 20:49:54'

    time_str = '2009-12-11 08:54:37'

    time_str = '2009-12-11 13:51:07'
    time_str = '2009-12-11 13:50:58'
    time_str = '2009-12-11 13:51:16'

    time_str = '2009-12-11 14:30:06'

    time_str = '2009-12-11 22:38:25'


    time_str = '2009-12-11 13:22:28'

    time_str = '2009-12-11 13:31:16'

    time_str = '2009-12-11 18:40:20'

    time_str = '2009-12-11 13:52:48'






    # res_str = get_activity_by_dnn({},{}, {}, time_str)
    # res_str = get_activity_by_dnn({},{}, {}, time_str, action='audio_vision')



    # print(res_str)
    # get_activity_duration_output()
    # exit(0)

    # action = 15
    # interval = ACTION_INTERVAL_DICT[action]
    # time_str = '2009-10-16 08:40:50'
    
    # action = check_motion_action(action, interval, time_str)
    # print("action:", action)

    # # # get_activity_duration()

    # exit(0)
    ### test   2009-10-16 - 2010-01-06, totally 84, ignore 01-06
    #base_date = '2009-12-11'

    #motion_dict = get_motion_activity(base_date)
    #print(len(motion_dict))
    #print(motion_dict)
    print(get_activity_duration_cnt_set())


    # base_date = '2009-10-16'

    # activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = get_activity_date(base_date)

    # print("==============")
    # print(len(activity_date_dict))
    # print(len(activity_begin_dict))
    # print(len(activity_end_dict))

    # print("begin dict")
    # print(activity_begin_dict)
    # print("end dict")
    # print(activity_end_dict)

    # exit(0)

    # get_day_begin_end(activity_date_dict, activity_begin_dict, activity_end_dict)


    # print(activity_date_dict.keys())

    # print(activity_date_dict)

    # activity_str, beginning_activity, end_activity = get_activity(activity_date_dict, activity_begin_list, activity_end_list, '2009-12-11 08:46:39' )


    # print(activity_str, beginning_activity, end_activity)

    # print(act, " begin:", activity_begin_dict[act], " end:", activity_end_dict[act])
