"""
Brief: Bayes model for activity probability, including vision, motion, audio-based model.
Author: Frank
Date: 07/10/2022
"""

import math
import tools_ascc
import copy
from datetime import datetime

PROB_OF_ALL_ACTIVITIES = {'Bed_to_Toilet': 0.039303482587064675, 'Morning_Meds': 0.01791044776119403, 'Watch_TV': 0.05124378109452736, 'Kitchen_Activity': 0.2517412935323383, 'Chores': 0.011442786069651741, 'Leave_Home': 0.10248756218905472, 'Read': 0.14477611940298507, 'Guest_Bathroom': 0.15074626865671642, 'Master_Bathroom': 0.1328358208955224, 'Desk_Activity': 0.021890547263681594, 'Eve_Meds': 0.007462686567164179, 'Meditate': 0.00845771144278607, 'Dining_Rm_Activity': 0.009950248756218905, 'Master_Bedroom_Activity': 0.04975124378109453}

READ_ACTIVITY = 'read'


# HMM Trans matrix, get it from motion_hmm.py
HMM_TRANS_MATRIX = {'Desk_Activity': {'Desk_Activity': 0.06451612903225806, 'Morning_Meds': 0.0, 'Leave_Home': 0.06451612903225806, 'Kitchen_Activity': 0.1935483870967742, 'Guest_Bathroom': 0.16129032258064516, 'Sleep': 0.0, 'Chores': 0.0967741935483871, 'Read': 0.0967741935483871, 'Master_Bathroom': 0.12903225806451613, 'Master_Bedroom_Activity': 0.0967741935483871, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.06451612903225806, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.03225806451612903}, 'Morning_Meds': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.2, 'Kitchen_Activity': 0.0, 'Guest_Bathroom': 0.24, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.48, 'Master_Bathroom': 0.0, 'Master_Bedroom_Activity': 0.04, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.04, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Leave_Home': {'Desk_Activity': 0.02702702702702703, 'Morning_Meds': 0.032432432432432434, 'Leave_Home': 0.17297297297297298, 'Kitchen_Activity': 0.21621621621621623, 'Guest_Bathroom': 0.10270270270270271, 'Sleep': 0.021621621621621623, 'Chores': 0.005405405405405406, 'Read': 0.2810810810810811, 'Master_Bathroom': 0.06486486486486487, 'Master_Bedroom_Activity': 0.032432432432432434, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.02702702702702703, 'Meditate': 0.016216216216216217, 'Dining_Rm_Activity': 0.0}, 'Kitchen_Activity': {'Desk_Activity': 0.010554089709762533, 'Morning_Meds': 0.026385224274406333, 'Leave_Home': 0.12928759894459102, 'Kitchen_Activity': 0.08443271767810026, 'Guest_Bathroom': 0.22163588390501318, 'Sleep': 0.005277044854881266, 'Chores': 0.0079155672823219, 'Read': 0.23746701846965698, 'Master_Bathroom': 0.12137203166226913, 'Master_Bedroom_Activity': 0.0395778364116095, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0158311345646438, 'Watch_TV': 0.08179419525065963, 'Meditate': 0.005277044854881266, 'Dining_Rm_Activity': 0.013192612137203167}, 'Guest_Bathroom': {'Desk_Activity': 0.01809954751131222, 'Morning_Meds': 0.0, 'Leave_Home': 0.12217194570135746, 'Kitchen_Activity': 0.3393665158371041, 'Guest_Bathroom': 0.08597285067873303, 'Sleep': 0.00904977375565611, 'Chores': 0.00904977375565611, 'Read': 0.1493212669683258, 'Master_Bathroom': 0.10407239819004525, 'Master_Bedroom_Activity': 0.10407239819004525, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.058823529411764705, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Sleep': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.06666666666666667, 'Kitchen_Activity': 0.26666666666666666, 'Guest_Bathroom': 0.0, 'Sleep': 0.0, 'Chores': 0.06666666666666667, 'Read': 0.0, 'Master_Bathroom': 0.5333333333333333, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.06666666666666667, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Chores': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.23809523809523808, 'Kitchen_Activity': 0.09523809523809523, 'Guest_Bathroom': 0.09523809523809523, 'Sleep': 0.047619047619047616, 'Chores': 0.0, 'Read': 0.19047619047619047, 'Master_Bathroom': 0.23809523809523808, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.0, 'Meditate': 0.09523809523809523, 'Dining_Rm_Activity': 0.0}, 'Read': {'Desk_Activity': 0.03319502074688797, 'Morning_Meds': 0.004149377593360996, 'Leave_Home': 0.07883817427385892, 'Kitchen_Activity': 0.46473029045643155, 'Guest_Bathroom': 0.1908713692946058, 'Sleep': 0.004149377593360996, 'Chores': 0.02074688796680498, 'Read': 0.04979253112033195, 'Master_Bathroom': 0.07053941908713693, 'Master_Bedroom_Activity': 0.024896265560165973, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.04149377593360996, 'Meditate': 0.004149377593360996, 'Dining_Rm_Activity': 0.012448132780082987}, 'Master_Bathroom': {'Desk_Activity': 0.02857142857142857, 'Morning_Meds': 0.02857142857142857, 'Leave_Home': 0.15, 'Kitchen_Activity': 0.3357142857142857, 'Guest_Bathroom': 0.05, 'Sleep': 0.014285714285714285, 'Chores': 0.04285714285714286, 'Read': 0.15, 'Master_Bathroom': 0.1, 'Master_Bedroom_Activity': 0.05, 'Bed_to_Toilet': 0.007142857142857143, 'Eve_Meds': 0.0, 'Watch_TV': 0.02142857142857143, 'Meditate': 0.014285714285714285, 'Dining_Rm_Activity': 0.007142857142857143}, 'Master_Bedroom_Activity': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.12698412698412698, 'Kitchen_Activity': 0.20634920634920634, 'Guest_Bathroom': 0.031746031746031744, 'Sleep': 0.031746031746031744, 'Chores': 0.0, 'Read': 0.19047619047619047, 'Master_Bathroom': 0.3968253968253968, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.015873015873015872, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Bed_to_Toilet': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.0, 'Kitchen_Activity': 0.0, 'Guest_Bathroom': 0.0, 'Sleep': 1.0, 'Chores': 0.0, 'Read': 0.0, 'Master_Bathroom': 0.0, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.0, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Eve_Meds': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.0, 'Kitchen_Activity': 0.2857142857142857, 'Guest_Bathroom': 0.42857142857142855, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.0, 'Master_Bathroom': 0.14285714285714285, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.14285714285714285, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Watch_TV': {'Desk_Activity': 0.04054054054054054, 'Morning_Meds': 0.0, 'Leave_Home': 0.05405405405405406, 'Kitchen_Activity': 0.40540540540540543, 'Guest_Bathroom': 0.3108108108108108, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.013513513513513514, 'Master_Bathroom': 0.08108108108108109, 'Master_Bedroom_Activity': 0.02702702702702703, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.013513513513513514, 'Watch_TV': 0.04054054054054054, 'Meditate': 0.013513513513513514, 'Dining_Rm_Activity': 0.0}, 'Meditate': {'Desk_Activity': 0.06666666666666667, 'Morning_Meds': 0.0, 'Leave_Home': 0.3333333333333333, 'Kitchen_Activity': 0.2, 'Guest_Bathroom': 0.0, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.06666666666666667, 'Master_Bathroom': 0.0, 'Master_Bedroom_Activity': 0.06666666666666667, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.06666666666666667, 'Meditate': 0.2, 'Dining_Rm_Activity': 0.0}, 'Dining_Rm_Activity': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.18181818181818182, 'Kitchen_Activity': 0.36363636363636365, 'Guest_Bathroom': 0.0, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.0, 'Master_Bathroom': 0.09090909090909091, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.18181818181818182, 'Meditate': 0.09090909090909091, 'Dining_Rm_Activity': 0.09090909090909091}}
# HMM Start matrix
HMM_START_MATRIX = {'Desk_Activity': 0.0, 'Morning_Meds': 0.09090909090909091, 'Leave_Home': 0.11363636363636363, 'Kitchen_Activity': 0.25, 'Guest_Bathroom': 0.11363636363636363, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.0, 'Master_Bathroom': 0.4090909090909091, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.022727272727272728, 'Eve_Meds': 0.0, 'Watch_TV': 0.0, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}


ACTIVITY_BED_TO_TOILET  = 'Bed_to_Toilet'
ACTIVITY_MORNING_MEDS = 'Morning_Meds'
ACTIVITY_WATCH_TV = 'Watch_TV'
ACTIVITY_KITCHEN = 'Kitchen_Activity'
ACTIVITY_CHORES = 'Chores'
ACTIVITY_LEAVE_HOME = 'Leave_Home'
ACTIVITY_READ = 'Read'
ACTIVITY_GUEST_BATHROOM = 'Guest_Bathroom'
ACTIVITY_MASTER_BATHROOM = 'Master_Bathroom'
ACTIVITY_DESK_ACTIVITY = 'Desk_Activity'
ACTIVITY_EVE_MEDS = 'Eve_Meds'
ACTIVITY_MEDITATE = 'Meditate'
ACTIVITY_DINING_RM_ACTIVITY = 'Dining_Rm_Activity'
ACTIVITY_MASTER_BEDROOM = 'Master_Bedroom_Activity'
ACTIVITY_SLEEP = 'Sleep'


ACTIVITY_DICT = {
ACTIVITY_BED_TO_TOILET : 'Bed_to_Toilet',
ACTIVITY_MORNING_MEDS : 'Morning_Meds',
ACTIVITY_WATCH_TV : 'Watch_TV',
ACTIVITY_KITCHEN : 'Kitchen_Activity',
ACTIVITY_CHORES : 'Chores',
ACTIVITY_LEAVE_HOME : 'Leave_Home',
ACTIVITY_READ : 'Read',
ACTIVITY_GUEST_BATHROOM : 'Guest_Bathroom',
ACTIVITY_MASTER_BATHROOM : 'Master_Bathroom',
ACTIVITY_DESK_ACTIVITY : 'Desk_Activity',
ACTIVITY_EVE_MEDS : 'Eve_Meds',
ACTIVITY_MEDITATE : 'Meditate',
ACTIVITY_DINING_RM_ACTIVITY : 'Dining_Rm_Activity',
ACTIVITY_MASTER_BEDROOM : 'Master_Bedroom_Activity'
}

# Location
LOCATION_READINGROOM = 'readingroom'
LOCATION_BATHROOM = 'bathroom'
LOCATION_BEDROOM = 'bedroom'
LOCATION_LIVINGROOM = 'livingroom'
LOCATION_KITCHEN = 'Kitchen'
LOCATION_DININGROOM = 'diningroom'
LOCATION_DOOR = 'door'
LOCATION_LOBBY = 'lobby'
OTHER = 'other'

P1_Location_Under_Act = {
ACTIVITY_BED_TO_TOILET: {LOCATION_BATHROOM:0.99},
ACTIVITY_MORNING_MEDS : {LOCATION_KITCHEN: 0.6, LOCATION_BEDROOM: 0.4},
ACTIVITY_WATCH_TV : {LOCATION_LIVINGROOM: 0.99},
ACTIVITY_KITCHEN : {LOCATION_KITCHEN: 0.99},
ACTIVITY_CHORES : {LOCATION_BATHROOM: 0.1, LOCATION_LIVINGROOM: 0.2, LOCATION_KITCHEN: 0.3, LOCATION_LOBBY: 0.1, LOCATION_BEDROOM: 0.2, LOCATION_DININGROOM:0.1},
ACTIVITY_LEAVE_HOME : {LOCATION_DOOR: 0.9, LOCATION_LOBBY: 0.1},
ACTIVITY_READ :{LOCATION_LIVINGROOM:0.95},
ACTIVITY_GUEST_BATHROOM : {LOCATION_BATHROOM: 0.99},
ACTIVITY_MASTER_BATHROOM : {LOCATION_BATHROOM: 0.99},
ACTIVITY_DESK_ACTIVITY : {LOCATION_LIVINGROOM:0.9},
ACTIVITY_EVE_MEDS : {LOCATION_KITCHEN: 0.6, LOCATION_BEDROOM: 0.4},
ACTIVITY_MEDITATE : {LOCATION_KITCHEN: 0.7, LOCATION_BEDROOM: 0.3},
ACTIVITY_DINING_RM_ACTIVITY : {LOCATION_DININGROOM: 0.99},
ACTIVITY_MASTER_BEDROOM : {LOCATION_BEDROOM: 0.99}

}


# Motion
MOTION_TYPE_SITTING = 'sitting'
MOTION_TYPE_STANDING = 'standing'
MOTION_TYPE_WALKING = 'walking'
MOTION_TYPE_SQUATING = 'squating' # treat it as sitting or ignore it 
MOTION_TYPE_LAYING = 'laying'

P2_Motion_type_Under_Act = {
ACTIVITY_BED_TO_TOILET: {MOTION_TYPE_SITTING:0.3, MOTION_TYPE_STANDING:0.6, MOTION_TYPE_WALKING:0.1},
ACTIVITY_MORNING_MEDS : {MOTION_TYPE_SITTING:0.8, MOTION_TYPE_STANDING:0.2, MOTION_TYPE_WALKING:0.0001},
ACTIVITY_WATCH_TV :{MOTION_TYPE_SITTING:0.8, MOTION_TYPE_STANDING:0.1, MOTION_TYPE_WALKING:0.1},
ACTIVITY_KITCHEN : {MOTION_TYPE_SITTING:0.5, MOTION_TYPE_STANDING:0.3, MOTION_TYPE_WALKING:0.2},
ACTIVITY_CHORES : {MOTION_TYPE_SITTING:0.0001, MOTION_TYPE_STANDING:0.7, MOTION_TYPE_WALKING:0.3},
ACTIVITY_LEAVE_HOME : {MOTION_TYPE_STANDING:0.7, MOTION_TYPE_WALKING:0.3},
ACTIVITY_READ : {MOTION_TYPE_SITTING:0.9, MOTION_TYPE_STANDING:0.1, MOTION_TYPE_WALKING:0.0001},
ACTIVITY_GUEST_BATHROOM : {MOTION_TYPE_SITTING:0.3, MOTION_TYPE_STANDING:0.6, MOTION_TYPE_WALKING:0.1},
ACTIVITY_MASTER_BATHROOM : {MOTION_TYPE_SITTING:0.3, MOTION_TYPE_STANDING:0.6, MOTION_TYPE_WALKING:0.1},
ACTIVITY_DESK_ACTIVITY : {MOTION_TYPE_SITTING:0.95, MOTION_TYPE_STANDING:0.05, MOTION_TYPE_WALKING:0.0001},
ACTIVITY_EVE_MEDS : {MOTION_TYPE_SITTING:0.8, MOTION_TYPE_STANDING:0.2, MOTION_TYPE_WALKING:0.0001},
ACTIVITY_MEDITATE : {MOTION_TYPE_SITTING:0.2, MOTION_TYPE_STANDING:0.8, MOTION_TYPE_WALKING:0.0001},
ACTIVITY_DINING_RM_ACTIVITY : {MOTION_TYPE_SITTING:0.7, MOTION_TYPE_STANDING:0.2, MOTION_TYPE_WALKING:0.1},
ACTIVITY_MASTER_BEDROOM : {MOTION_TYPE_SITTING:0.3, MOTION_TYPE_STANDING:0.1, MOTION_TYPE_WALKING:0.2, MOTION_TYPE_LAYING:0.4}

}

# Audio
AUDIO_TYPE_ENV = 'quite'
AUDIO_TYPE_DOOR = 'door_open_closed'
AUDIO_TYPE_EATING = 'eating'	
AUDIO_TYPE_KEYBOARD = 'keyboard'	
AUDIO_TYPE_POURING_WATER_INTO_GLASS = 'pouring_water_into_glass'	
AUDIO_TYPE_TOOTHBRUSHING = 'toothbrushing'	
AUDIO_TYPE_VACUUM = 'vacuum'	
AUDIO_TYPE_DRINKING = 'drinking'	
AUDIO_TYPE_FLUSH_TOILET = 'flush_toilet'	
AUDIO_TYPE_MICROWAVE = 'microwave'	
AUDIO_TYPE_TV = 'tv_news'	
AUDIO_TYPE_WASHING_HAND = 'washing_hand'
AUDIO_TYPE_QUITE = AUDIO_TYPE_ENV

P3_Audio_type_Under_Act = {
ACTIVITY_BED_TO_TOILET: {AUDIO_TYPE_FLUSH_TOILET:0.2, AUDIO_TYPE_ENV:0.4, AUDIO_TYPE_WASHING_HAND:0.4},
ACTIVITY_MORNING_MEDS : {AUDIO_TYPE_POURING_WATER_INTO_GLASS:0.1, AUDIO_TYPE_DRINKING:0.2, AUDIO_TYPE_ENV:0.7},
ACTIVITY_WATCH_TV : {AUDIO_TYPE_ENV: 0.1, AUDIO_TYPE_TV: 0.9},
ACTIVITY_KITCHEN : {AUDIO_TYPE_EATING: 0.3, AUDIO_TYPE_POURING_WATER_INTO_GLASS:0.1, AUDIO_TYPE_DRINKING:0.2, AUDIO_TYPE_MICROWAVE:0.1, AUDIO_TYPE_ENV:0.3},
ACTIVITY_CHORES : {AUDIO_TYPE_VACUUM:0.9, AUDIO_TYPE_ENV:0.1},
ACTIVITY_LEAVE_HOME : {AUDIO_TYPE_DOOR: 0.8, AUDIO_TYPE_ENV:0.2},
ACTIVITY_READ : {AUDIO_TYPE_ENV: 0.99},
ACTIVITY_GUEST_BATHROOM : {AUDIO_TYPE_TOOTHBRUSHING : 0.2, AUDIO_TYPE_FLUSH_TOILET: 0.1, AUDIO_TYPE_ENV:0.5, AUDIO_TYPE_WASHING_HAND:0.2},
ACTIVITY_MASTER_BATHROOM : {AUDIO_TYPE_TOOTHBRUSHING : 0.2, AUDIO_TYPE_FLUSH_TOILET: 0.1, AUDIO_TYPE_ENV:0.5, AUDIO_TYPE_WASHING_HAND:0.2},
ACTIVITY_DESK_ACTIVITY : {AUDIO_TYPE_KEYBOARD:0.3, AUDIO_TYPE_ENV: 0.7},
ACTIVITY_EVE_MEDS : {AUDIO_TYPE_POURING_WATER_INTO_GLASS:0.1, AUDIO_TYPE_DRINKING:0.2, AUDIO_TYPE_ENV:0.7},
ACTIVITY_MEDITATE : {AUDIO_TYPE_ENV:0.99},
ACTIVITY_DINING_RM_ACTIVITY : {AUDIO_TYPE_EATING: 0.4, AUDIO_TYPE_POURING_WATER_INTO_GLASS:0.1, AUDIO_TYPE_DRINKING:0.2, AUDIO_TYPE_ENV:0.3},
ACTIVITY_MASTER_BEDROOM : {AUDIO_TYPE_ENV:0.99}
}


# Object,book, medicine, laptop, plate & fork & food, toilet
OBJECT_BOOK = 'book'
OBJECT_MEDICINE = 'medicine'
OBJECT_LAPTOP = 'laptop'
OBJECT_PLATE = 'plate'
OBJECT_FORK = 'fork'
OBJECT_FOOD = 'food'
OBJECT_TOILET = 'toilet'
OBJECT_MICROWAVE = 'microwave'
OBJECT_REFRIGERATOR = 'refrigerator'
OBJECT_VACUUM = 'vacuum'
OBJECT_PAPER = 'paper'
OBJECT_DOOR = 'door'
OBJECT_TV = 'tv'

P4_Object_Under_Act = {
ACTIVITY_BED_TO_TOILET: {OBJECT_TOILET:0.8},
ACTIVITY_MORNING_MEDS : {OBJECT_MEDICINE:0.8, OBJECT_MICROWAVE:0.1},
ACTIVITY_WATCH_TV : {OBJECT_TV:0.9},
ACTIVITY_KITCHEN : {OBJECT_PLATE:0.7, OBJECT_FORK:0.7, OBJECT_FOOD:0.7, OBJECT_MICROWAVE:0.3, OBJECT_REFRIGERATOR:0.2},
ACTIVITY_CHORES : {OBJECT_TOILET:0.1, OBJECT_VACUUM:0.7, OBJECT_MICROWAVE:0.1},
ACTIVITY_LEAVE_HOME : {OBJECT_DOOR:0.9},
ACTIVITY_READ : {OBJECT_BOOK:0.9, OBJECT_MICROWAVE:0.1},
ACTIVITY_GUEST_BATHROOM : {OBJECT_TOILET:0.7},
ACTIVITY_MASTER_BATHROOM : {OBJECT_TOILET:0.7},
ACTIVITY_DESK_ACTIVITY : {OBJECT_BOOK:0.3, OBJECT_LAPTOP:0.4, OBJECT_PAPER:0.2, OBJECT_MICROWAVE:0.1},
ACTIVITY_EVE_MEDS : {OBJECT_MEDICINE:0.8, OBJECT_MICROWAVE:0.1},
ACTIVITY_MEDITATE : {OBJECT_MICROWAVE:0.7, OBJECT_REFRIGERATOR:0.5},
ACTIVITY_DINING_RM_ACTIVITY : {OBJECT_PLATE:0.8, OBJECT_FORK:0.8, OBJECT_FOOD:0.8}, 
ACTIVITY_MASTER_BEDROOM : {OBJECT_MEDICINE:0.2}

}

MIN_Prob = 1e-150

act_duration_cnt_dict = tools_ascc.get_activity_duration_cnt_set()

DURATION_PROB_ALPHA = 10

def get_end_of_activity_prob_by_duration(activity_duration, activity):
    d_lis = act_duration_cnt_dict[activity]
    total_cnt = len(d_lis)
    cnt = 0
    for d in d_lis:
        if activity_duration >= d:
            cnt += 1
    if cnt == 0:
        cnt = 1

    prob = 1 - cnt * 1.0 /total_cnt

    # prob = prob * 10
    # # prob = math.pow(10, int(prob)) * MIN_Prob
    # prob = prob * MIN_Prob
    
    if prob < 0.01:
        prob = MIN_Prob

    return prob


def get_activity_type(cur_time_str):
    
    act_type_list = ['M', 'A', 'N']
    cur_type_time = datetime.strptime(cur_time_str.split()[1], tools_ascc.HOUR_TIME_FORMAT)

    if cur_type_time.hour < tools_ascc.ACTIVITY_NOON_HOUR - 2:
        act_type_list = ['M']
    elif (cur_type_time.hour >= tools_ascc.ACTIVITY_NOON_HOUR - 2) and (cur_type_time.hour < tools_ascc.ACTIVITY_NOON_HOUR + 2):
        act_type_list = ['M', 'A']
    elif (cur_type_time.hour >= tools_ascc.ACTIVITY_NOON_HOUR + 2) and (cur_type_time.hour < tools_ascc.ACTIVITY_NIGHT_HOUR - 2):
        act_type_list = ['A']
    elif (cur_type_time.hour >= tools_ascc.ACTIVITY_NIGHT_HOUR - 2) and (cur_type_time.hour < tools_ascc.ACTIVITY_NIGHT_HOUR + 2):
        act_type_list = ['A', 'N']
    else:
        act_type_list = ['N']

    return act_type_list

#prob_of_location_under_all_acts
Prob_Of_Location_Under_All_Act = {}


# for image recognition, we can get the reuslt for DNN, from the confusion matrix
DNN_ACC_IMAGE = 0.99

CNN_ACC_MOTION = 0.93
CNN_ACC_AUDIO = 0.92 # Sound-Recognition-Tutorial/train_res.txt
CNN_ACC_SOUND = CNN_ACC_AUDIO

YOLO_ACC_OBJECT = 0.9

TOTAL_ACTIVITY_CNT = len(PROB_OF_ALL_ACTIVITIES)


# CNN Confusion matrix

DATE_HOUR_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


class Bayes_Model_Vision_Location(object):
    """
    This class is an implementation of the Bayes Model.
    """

    def __init__(self, hmm_model, simulation = False, base_date = '2009-12-11'):
        self.hmm_model = hmm_model
        self.simulation = simulation
        self.cur_time = ''

        self.base_date = base_date

        self.activity_dict_init()


    def activity_dict_init(self):

        date_day_hour_time = self.base_date

        self.activity_date_dict, self.activity_begin_dict, self.activity_end_dict, \
        self.activity_begin_list, self.activity_end_list  = tools_ascc.get_activity_date(date_day_hour_time)

    def set_time(self, time):
        self.cur_time = time

    def set_base_date(self, date_time):
        self.base_date = date_time #base_date = '2009-12-11'

    def set_act_name(self, act_name):
        self.act_name = act_name

    def set_location(self, location):
        self.location = location

    def set_location_prob(self, prob):
        self.location_prob = prob

    def get_prob(self, pre_activity_list, act_name, location, act_duration, mode = None):
        """ Return the state set of this model. """
        p = 0

        p =  self.prob_of_location_under_act(location, act_name) \
             /(self.prob_of_location_under_all_acts(location)) \
            * self.prob_of_location_using_vision(location, act_name)

        if mode == 'HMM':
            p =  self.prob_of_location_under_act(location, act_name) \
             * self.prob_prior_act_by_prelist(pre_activity_list, act_name, act_duration) /(self.prob_of_location_under_all_acts(location)) \
                * self.prob_of_location_using_vision(location, act_name)

        return p


    def prob_of_location_under_act(self, location, act_name):
        p = MIN_Prob

        try:
            p = P1_Location_Under_Act[act_name][location]
        except Exception as e:
            pass
            # print('Got error from P1_Location_Under_Act, location, act_name:', location, ', ', act_name, ', err:', e)
        
        # print('prob_of_location_under_act: location, act_name, p:', location, ' ', act_name, ' ', p)
        return p

    # def prob_prior_act(self, pre_activity, act_name):
    #     p = MIN_Prob
    #     try:
    #         p = HMM_TRANS_MATRIX[pre_activity][act_name]
    #     except Exception as e:
    #         print('Got error from HMM_TRANS_MATRIX, pre_activity, act_name:', pre_activity, ', ', act_name, ', err:', e)

    #     # during acitivty
    #     if pre_activity == act_name:
    #         p = 1
    #         pass

    #     return p

    def prob_prior_act_by_prelist(self, pre_activity_list, target_act_name, duration = None):
        p = MIN_Prob

        pre_act_list = copy.deepcopy(pre_activity_list)

        if len(pre_act_list) == 0:
            return HMM_START_MATRIX[target_act_name]

        if pre_act_list[-1] == target_act_name:

            p = get_end_of_activity_prob_by_duration(duration, target_act_name)

            #print('prob_prior_act_by_prelist == target_act_name, duration, p', target_act_name, ' ', duration, ' ', p)

            return p

        act_type_list = get_activity_type(self.cur_time)
        for type in act_type_list:
            activity_type = type
            node = tools_ascc.Activity_Node_Observable(target_act_name, activity_type, 0)
            next_act = node.activity_res_generation()

            if pre_act_list[-1] == next_act:
                p = get_end_of_activity_prob_by_duration(duration, target_act_name)

                #print('prob_prior_act_by_prelist == next_act, duration, p', next_act, ' ', duration, ' ', p)

                return p

        
        res = {}
        act_type_list = get_activity_type(self.cur_time)
        
        for index in tools_ascc.ACTIVITY_DICT.keys():
            act = tools_ascc.ACTIVITY_DICT[index]
            for type in act_type_list:
                activity_type = type
                node = tools_ascc.Activity_Node_Observable(act, activity_type, 0)
        
                next_act = node.activity_res_generation()

                test_lis = copy.deepcopy(pre_act_list)
                test_lis.append(next_act)
                prob = self.hmm_model.evaluate(test_lis)

                # print('test_lis:', test_lis)
                # print('prob:', prob)
                res[next_act] = prob


        # print('=========================================================')

        sd = sorted(res.items(), key=tools_ascc.sorter_take_count, reverse=True)
        #print(sd)

        for k, v in sd:
            for type in act_type_list:
                activity_type = type
                # tmp_node = tools_ascc.Activity_Node_Observable(k, activity_type, 0)
                # tmp_act = tmp_node.activity_res_generation()
                tmp_act = k

                target_node = tools_ascc.Activity_Node_Observable(target_act_name, activity_type, 0)
                target_act = target_node.activity_res_generation()

                if target_act == tmp_act:
                    p = v
                    break

        #print('prob_prior_act_by_prelist target_act_name, duration, p', target_act_name, ' ', duration, ' ', p)

        
        return p


    # Total probability rule, 15 activities
    def prob_of_location_under_all_acts(self, location):
        p = MIN_Prob

        for act in PROB_OF_ALL_ACTIVITIES.keys():
            p = p + self.prob_of_location_under_act(location, act) * self.prob_of_activity_in_dataset(act)

        # print('prob_of_location_under_all_acts, location, p:', location, ' ', p)

        return p
    
    # From CNN model, confusion matrix for simulation
    # For real time experiments, use CNN to predict the activity and get the probability
    def prob_of_location_using_vision(self, location, act = None):
        p = MIN_Prob
        # todo, how to get the confusion matrix of CNN recognition model

        if self.simulation == True:
            activity, _, _ = self.get_activity_from_dataset_by_time(self.cur_time)
            if activity == act:
                p = DNN_ACC_IMAGE
            else:
                p = (1-DNN_ACC_IMAGE)/(TOTAL_ACTIVITY_CNT-1) # totally 15 activities

        else:
            # find the probability of location from CNN recognition results
            # vision_res_activity_list = tools_ascc.get_activity_by_vision_dnn(time_str, action='vision')
            # location = ''
            # prob = 0
            # if len(vision_res_activity_list) > 0:
            #     res = vision_res_activity_list[0]
            #     # kitchen(0.99791986)     kitchen(0.9997546)      kitchen(0.99589807)     kitchen(0.99955696)     kitchen(0.92794806)     
            #     # kitchen(0.9657237)      kitchen(0.99801105)     kitchen(0.99981874)   kitchen(0.99988997)      kitchen(0.97507715)
            #     location = res.split('(')[0]
            #     prob = res.split('(')[1].split(')')[0]

            p = self.location_prob

            pass

        #print('prob_of_location_using_vision:, location, act, p:', act, ' ', location, ' ', p)
        return p

    def prob_of_activity_in_dataset(self, act):
        "Get from dataset"
        p = PROB_OF_ALL_ACTIVITIES[act]

        return p



    def get_activity_from_dataset_by_time(self, time_str):

        # base_date = MILAN_BASE_DATE
        # day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days=i + 56)
        # day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        # activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
        #     tools_ascc.get_activity_date(day_time_str)

        # # get expected_activity at current time (running time)
        # run_time = self.running_time.strftime(DATE_HOUR_TIME_FORMAT)

        test_time_str = '2009-12-11 12:57:20'
        # time_str = test_time_str
        expected_activity_str, expected_beginning_activity, expected_end_activity = \
            tools_ascc.get_activity(self.activity_date_dict, self.activity_begin_list,
                                    self.activity_end_list, time_str)
                
        return expected_activity_str, expected_beginning_activity, expected_end_activity

# Bayes Model using Motion
# todo
class Bayes_Model_Motion(object):
    """
    This class is an implementation of the Bayes Model.
    """

    def __init__(self, hmm_model, simulation = False, base_date = '2009-12-11'):
        self.hmm_model = hmm_model
        self.simulation = simulation
        self.cur_time = ''

        self.base_date = base_date

        self.activity_dict_init()


    def activity_dict_init(self):

        date_day_hour_time = self.base_date + " 00:00:00"

        self.activity_date_dict, self.activity_begin_dict, self.activity_end_dict, \
        self.activity_begin_list, self.activity_end_list  = tools_ascc.get_activity_date(date_day_hour_time)

    def set_time(self, time):
        self.cur_time = time

    def set_base_date(self, date_time):
        self.base_date = date_time #base_date = '2009-12-11'

    def set_act_name(self, act_name):
        self.act_name = act_name

    def set_motion_type(self, motion_type):
        self.motion_type = motion_type

    def set_motion_type_prob(self, prob):
        self.motion_type_prob = prob

    def get_prob(self, pre_activity, act_name, motion_type, activity_duration, mode = None):
        """ Return the state set of this model. """
        p = 0
        p =  self.prob_of_motion_type_under_act(motion_type, act_name) \
            /(self.prob_of_motion_type_under_all_acts(motion_type))\
                 * self.prob_of_motion_type_using_motion(motion_type, act_name)

        if mode == 'HMM':
            p =  self.prob_of_motion_type_under_act(motion_type, act_name) \
                * self.prob_prior_act_by_prelist(pre_activity, act_name, activity_duration) /(self.prob_of_motion_type_under_all_acts(motion_type))\
                    * self.prob_of_motion_type_using_motion(motion_type, act_name)


        return p


    def prob_of_motion_type_under_act(self, motion_type, act_name):
        p = MIN_Prob

        try:
            p = P2_Motion_type_Under_Act[act_name][motion_type]
        except Exception as e:
            pass
            # print('Got error from P2_Motion_type_Under_Act, motion, act_name:', motion_type, ', ', act_name, ', err:', e)

        # print('prob_of_motion_type_under_act motion_type, act_name, p:', motion_type, ' ', act_name, ' ', p)
        return p

    # def prob_prior_act(self, pre_activity, act_name):
    #     p = MIN_Prob
    #     try:
    #         p = HMM_TRANS_MATRIX[pre_activity][act_name]
    #     except Exception as e:
    #         pass
    #         # print('Got error from HMM_TRANS_MATRIX, pre_activity, act_name:', pre_activity, ', ', act_name, ', err:', e)

    #     # during acitivty
    #     if pre_activity == act_name:
    #         p = 1
    #         pass

    #     return p

    def prob_prior_act_by_prelist(self, pre_activity_list, target_act_name, duration = None):
        p = MIN_Prob

        pre_act_list = copy.deepcopy(pre_activity_list)
        #print('motion pre_act_list:', pre_act_list)

        if len(pre_act_list) == 0:
            return HMM_START_MATRIX[target_act_name]

        if pre_act_list[-1] == target_act_name:

            p = get_end_of_activity_prob_by_duration(duration, target_act_name)
            #print('prob_prior_act_by_prelist target_act_name, == duration, p:', target_act_name, ' ', duration, ' ', p)

            return p
        
        act_type_list = get_activity_type(self.cur_time)
        for type in act_type_list:
            activity_type = type
            node = tools_ascc.Activity_Node_Observable(target_act_name, activity_type, 0)
            next_act = node.activity_res_generation()

            if pre_act_list[-1] == next_act:
                p = get_end_of_activity_prob_by_duration(duration, target_act_name)

                #print('prob_prior_act_by_prelist == next_act, duration, p', next_act, ' ', duration, ' ', p)

                return p


        res = {}
        act_type_list = get_activity_type(self.cur_time)

        
        for index in tools_ascc.ACTIVITY_DICT.keys():
            act = tools_ascc.ACTIVITY_DICT[index]
            for type in act_type_list:
                activity_type = type
                node = tools_ascc.Activity_Node_Observable(act, activity_type, 0)
        
                next_act = node.activity_res_generation()

                test_lis = copy.deepcopy(pre_act_list)
                test_lis.append(next_act)
                prob = self.hmm_model.evaluate(test_lis)

                # print('motion test_lis:', test_lis)
                #print('next_act, prob:', next_act, ' ', prob)
                res[next_act] = prob

        # print('=========================================================')

        sd = sorted(res.items(), key=tools_ascc.sorter_take_count, reverse=True)
        #print(sd)

        for k, v in sd:
            for type in act_type_list:
                activity_type = type
                # tmp_node = tools_ascc.Activity_Node_Observable(k, activity_type, 0)
                # tmp_act = tmp_node.activity_res_generation()
                tmp_act = k

                target_node = tools_ascc.Activity_Node_Observable(target_act_name, activity_type, 0)
                target_act = target_node.activity_res_generation()

                if target_act == tmp_act:
                    p = v
                    break
        
        #print('prob_prior_act_by_prelist target_act_name, duration, p:', target_act_name, ' ', duration, ' ', p)

        return p

    # Total probability rule, 15 activities
    def prob_of_motion_type_under_all_acts(self, motion_type):
        p = MIN_Prob

        for act in PROB_OF_ALL_ACTIVITIES.keys():
            p = p + self.prob_of_motion_type_under_act(motion_type, act) * self.prob_of_activity_in_dataset(act)

        #print('prob_of_motion_type_under_all_acts motion_type, p:', motion_type, ' ', p)

        return p
    
    # From CNN model, confusion matrix for simulation
    # For real time experiments, use CNN to predict the activity and get the probability
    def prob_of_motion_type_using_motion(self, motion_type, act = None):
        p = MIN_Prob
        # todo, how to get the confusion matrix of CNN recognition model

        if self.simulation == True:
            activity, _, _ = self.get_activity_from_dataset_by_time(self.cur_time)
            if activity == act:
                p = CNN_ACC_MOTION
            else:
                p = (1-CNN_ACC_MOTION)/(TOTAL_ACTIVITY_CNT-1) # totally 15 activities

        else:
            # find the probability of location from CNN recognition results
            p = self.motion_type_prob
            pass

        #print('prob_of_motion_type_using_motion motion_type, act, p:', motion_type, ' ', act, ' ', p)

        return p

    def prob_of_activity_in_dataset(self, act):
        "Get from dataset"
        p = PROB_OF_ALL_ACTIVITIES[act]

        return p



    def get_activity_from_dataset_by_time(self, time_str):

        # base_date = MILAN_BASE_DATE
        # day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days=i + 56)
        # day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        # activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
        #     tools_ascc.get_activity_date(day_time_str)

        # # get expected_activity at current time (running time)
        # run_time = self.running_time.strftime(DATE_HOUR_TIME_FORMAT)

        test_time_str = '2009-12-11 12:58:33'
        # time_str = test_time_str
        expected_activity_str, expected_beginning_activity, expected_end_activity = \
            tools_ascc.get_activity(self.activity_date_dict, self.activity_begin_list,
                                    self.activity_end_list, time_str)
                
        return expected_activity_str, expected_beginning_activity, expected_end_activity




# Bayes Model using Audio
class Bayes_Model_Audio(object):
    """
    This class is an implementation of the Bayes Model.
    """

    def __init__(self, hmm_model, simulation = False, base_date = '2009-12-11'):
        self.hmm_model = hmm_model
        self.simulation = simulation
        self.cur_time = ''

        self.base_date = base_date

        self.activity_dict_init()


    def activity_dict_init(self):

        date_day_hour_time = self.base_date + " 00:00:00"

        self.activity_date_dict, self.activity_begin_dict, self.activity_end_dict, \
        self.activity_begin_list, self.activity_end_list  = tools_ascc.get_activity_date(date_day_hour_time)

    def set_time(self, time):
        self.cur_time = time

    def set_base_date(self, date_time):
        self.base_date = date_time #base_date = '2009-12-11'

    def set_act_name(self, act_name):
        self.act_name = act_name

    def set_audio_type(self, audio_type):
        self.audio_type = audio_type
    
    def set_audio_type_prob(self, prob):
        self.audio_type_prob = prob

    def get_prob(self, pre_activity, act_name, audio_type, activity_duration, mode=None):
        """ Return the state set of this model. """
        p = 0
        p =  self.prob_of_audio_type_under_act(audio_type, act_name) \
             /(self.prob_of_audio_type_under_all_acts(audio_type))\
                 * self.prob_of_audio_type_using_audio(audio_type, act_name)

        if mode == 'HMM':
            p =  self.prob_of_audio_type_under_act(audio_type, act_name) \
             * self.prob_prior_act_by_prelist(pre_activity, act_name, activity_duration) /(self.prob_of_audio_type_under_all_acts(audio_type))\
                 * self.prob_of_audio_type_using_audio(audio_type, act_name)        

        return p


    def prob_of_audio_type_under_act(self, audio_type, act_name):
        p = MIN_Prob

        try:
            p = P3_Audio_type_Under_Act[act_name][audio_type]
        except Exception as e:
            pass
            # print('Got error from P3_Audio_type_Under_Act, location, act_name:', audio_type, ', ', act_name, ', err:', e)
        
        # print('prob_of_audio_type_under_act audio_type, act_name, p:', audio_type, ' ', act_name, ' ', p)

        return p

    def prob_prior_act(self, pre_activity, act_name):
        p = MIN_Prob
        try:
            p = HMM_TRANS_MATRIX[pre_activity][act_name]
        except Exception as e:
            pass
            # print('Got error from HMM_TRANS_MATRIX, pre_activity, act_name:', pre_activity, ', ', act_name, ', err:', e)

        # during acitivty
        if pre_activity == act_name:
            p = 1
            pass

        return p

    def prob_prior_act_by_prelist(self, pre_activity_list, target_act_name, duration = None):
        p = MIN_Prob

        pre_act_list = copy.deepcopy(pre_activity_list)
        if len(pre_act_list) == 0:
            return HMM_START_MATRIX[target_act_name]

        if pre_act_list[-1] == target_act_name:

            p = get_end_of_activity_prob_by_duration(duration, target_act_name)
            #print('target_act_name, == duration, p:', target_act_name, ' ', duration, ' ', p)

            return p
        
        act_type_list = get_activity_type(self.cur_time)
        for type in act_type_list:
            activity_type = type
            node = tools_ascc.Activity_Node_Observable(target_act_name, activity_type, 0)
            next_act = node.activity_res_generation()

            if pre_act_list[-1] == next_act:
                p = get_end_of_activity_prob_by_duration(duration, target_act_name)

                # print('prob_prior_act_by_prelist == next_act, duration, p', next_act, ' ', duration, ' ', p)

                return p

        res = {}
        act_type_list = get_activity_type(self.cur_time)
        
        for index in tools_ascc.ACTIVITY_DICT.keys():
            act = tools_ascc.ACTIVITY_DICT[index]
            for type in act_type_list:
                activity_type = type
                node = tools_ascc.Activity_Node_Observable(act, activity_type, 0)
        
                next_act = node.activity_res_generation()

                test_lis = copy.deepcopy(pre_act_list)
                test_lis.append(next_act)
                prob = self.hmm_model.evaluate(test_lis)

                # print('test_lis:', test_lis)
                # print('prob:', prob)
                res[next_act] = prob

        # print('=========================================================')

        sd = sorted(res.items(), key=tools_ascc.sorter_take_count, reverse=True)
        #print('prob_prior_act_by_prelist HMM res:')
        #print(sd)

        for k, v in sd:
            for type in act_type_list:
                activity_type = type
                # tmp_node = tools_ascc.Activity_Node_Observable(k, activity_type, 0)
                # tmp_act = tmp_node.activity_res_generation()
                tmp_act = k

                target_node = tools_ascc.Activity_Node_Observable(target_act_name, activity_type, 0)
                target_act = target_node.activity_res_generation()

                if target_act == tmp_act:
                    p = v
                    break
        
        # print('prob_prior_act_by_prelist target_act_name, duration, p:', target_act_name, ' ', duration, ' ', p)

        return p

    # Total probability rule, 15 activities
    def prob_of_audio_type_under_all_acts(self, audio_type):
        p = MIN_Prob

        for act in PROB_OF_ALL_ACTIVITIES.keys():
            #print('act: ', act, ' audio type:', audio_type)
            #print(self.prob_of_audio_type_under_act(audio_type, act))
            #print(self.prob_of_activity_in_dataset(act))
            p = p + self.prob_of_audio_type_under_act(audio_type, act) * self.prob_of_activity_in_dataset(act)

        # print('prob_of_audio_type_under_all_acts prob_of_motion_type_under_all_acts audio_type, p:', audio_type, ' ', p)

        return p
    
    # From CNN model, confusion matrix for simulation
    # For real time experiments, use CNN to predict the activity and get the probability
    def prob_of_audio_type_using_audio(self, audio_type, act = None):
        p = MIN_Prob
        # todo, how to get the confusion matrix of CNN recognition model

        if self.simulation == True:
            activity, _, _ = self.get_activity_from_dataset_by_time(self.cur_time)
            if activity == act:
                p = CNN_ACC_AUDIO
            else:
                p = (1-CNN_ACC_AUDIO)/(TOTAL_ACTIVITY_CNT-1) # totally 15 activities

        else:
            # find the probability of location from CNN recognition results
            p = self.audio_type_prob
            pass

        # print('prob_of_audio_type_using_audio audio_type, act, p:', audio_type, ' ', act, ' ', p)

        return p

    def prob_of_activity_in_dataset(self, act):
        "Get from dataset"
        p = PROB_OF_ALL_ACTIVITIES[act]

        return p



    def get_activity_from_dataset_by_time(self, time_str):

        # base_date = MILAN_BASE_DATE
        # day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days=i + 56)
        # day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        # activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
        #     tools_ascc.get_activity_date(day_time_str)

        # # get expected_activity at current time (running time)
        # run_time = self.running_time.strftime(DATE_HOUR_TIME_FORMAT)

        test_time_str = '2009-12-11 12:58:33'
        # time_str = test_time_str
        expected_activity_str, expected_beginning_activity, expected_end_activity = \
            tools_ascc.get_activity(self.activity_date_dict, self.activity_begin_list,
                                    self.activity_end_list, time_str)
                
        return expected_activity_str, expected_beginning_activity, expected_end_activity



# Object bayes model
class Bayes_Model_Vision_Object(object):
    """
    This class is an implementation of the Bayes Model.
    """

    def __init__(self, hmm_model, simulation = False, base_date = '2009-12-11'):
        self.hmm_model = hmm_model
        self.simulation = simulation
        self.cur_time = ''

        self.base_date = base_date

        self.activity_dict_init()


    def activity_dict_init(self):

        date_day_hour_time = self.base_date + " 00:00:00"

        self.activity_date_dict, self.activity_begin_dict, self.activity_end_dict, \
        self.activity_begin_list, self.activity_end_list  = tools_ascc.get_activity_date(date_day_hour_time)

    def set_time(self, time):
        self.cur_time = time

    def set_base_date(self, date_time):
        self.base_date = date_time #base_date = '2009-12-11'

    def set_act_name(self, act_name):
        self.act_name = act_name

    def set_object(self, object):
        self.object = object

    def set_object_prob(self, prob):
        self.object_prob = prob

    def get_prob(self, pre_activity, act_name, object, activity_duration, mode=None):
        """ Return the state set of this model. """
        p = 0
        p =  self.prob_of_object_under_act(object, act_name) \
            /(self.prob_of_object_under_all_acts(object)) * self.prob_of_object_using_vision(object, act_name)

        if mode == 'HMM':
            p =  self.prob_of_object_under_act(object, act_name) \
                * self.prob_prior_act_by_prelist(pre_activity, act_name, activity_duration) /(self.prob_of_object_under_all_acts(object)) * self.prob_of_object_using_vision(object, act_name)
 
        return p


    def prob_of_object_under_act(self, object, act_name):
        p = MIN_Prob

        try:
            p = P4_Object_Under_Act[act_name][object]
        except Exception as e:
            pass
            # print('Got error from P4_Object_Under_Act, object, act_name:', object, ', ', act_name, ', err:', e)

        # print('prob_of_object_under_act, object, act_name, p:', object, ' ', act_name, ' ', p)
        return p

    def prob_prior_act(self, pre_activity, act_name):
        p = MIN_Prob
        try:
            p = HMM_TRANS_MATRIX[pre_activity][act_name] # todo: update a new method, using evaluate the sequenece, hmm model
        except Exception as e:
            pass
            # print('Got error from HMM_TRANS_MATRIX, pre_activity, act_name:', pre_activity, ', ', act_name, ', err:', e)

        # during acitivty
        if pre_activity == act_name:
            p = 1
            pass

        return p

    def prob_prior_act_by_prelist(self, pre_activity_list, target_act_name, duration = None):
        p = MIN_Prob

        pre_act_list = copy.deepcopy(pre_activity_list)

        if len(pre_act_list) == 0:
            return HMM_START_MATRIX[target_act_name]

        if pre_act_list[-1] == target_act_name:

            p = get_end_of_activity_prob_by_duration(duration, target_act_name)
            #print('prob_prior_act_by_prelist ==target_act_name, duration, p:', target_act_name, ' ', duration, ' ', p)

            return p

        act_type_list = get_activity_type(self.cur_time)
        for type in act_type_list:
            activity_type = type
            node = tools_ascc.Activity_Node_Observable(target_act_name, activity_type, 0)
            next_act = node.activity_res_generation()

            if pre_act_list[-1] == next_act:
                p = get_end_of_activity_prob_by_duration(duration, target_act_name)

                # print('prob_prior_act_by_prelist == next_act, duration, p', next_act, ' ', duration, ' ', p)

                return p

        res = {}
        act_type_list = get_activity_type(self.cur_time)
        
        for index in tools_ascc.ACTIVITY_DICT.keys():
            act = tools_ascc.ACTIVITY_DICT[index]
            for type in act_type_list:
                activity_type = type
                node = tools_ascc.Activity_Node_Observable(act, activity_type, 0)
        
                next_act = node.activity_res_generation()

                test_lis = copy.deepcopy(pre_act_list)
                test_lis.append(next_act)
                prob = self.hmm_model.evaluate(test_lis)

                # print('test_lis:', test_lis)
                # print('prob:', prob)
                res[next_act] = prob

        # print('=========================================================')

        sd = sorted(res.items(), key=tools_ascc.sorter_take_count, reverse=True)
        #print(sd)

        for k, v in sd:
            for type in act_type_list:
                activity_type = type
                # tmp_node = tools_ascc.Activity_Node_Observable(k, activity_type, 0)
                # tmp_act = tmp_node.activity_res_generation()
                tmp_act = k

                target_node = tools_ascc.Activity_Node_Observable(target_act_name, activity_type, 0)
                target_act = target_node.activity_res_generation()

                if target_act == tmp_act:
                    p = v
                    break
                
        # print('prob_prior_act_by_prelist target_act_name, duration, p:', target_act_name, ' ', duration, ' ', p)
    
        return p

    # Total probability rule, 15 activities
    def prob_of_object_under_all_acts(self, object):
        p = MIN_Prob

        for act in PROB_OF_ALL_ACTIVITIES.keys():
            p = p + self.prob_of_object_under_act(object, act) * self.prob_of_activity_in_dataset(act)

        # print('prob_of_object_under_all_acts objec, p:', object, ' ', p)
        return p
    
    # From CNN model, confusion matrix for simulation
    # For real time experiments, use CNN to predict the activity and get the probability
    def prob_of_object_using_vision(self, object, act = None):
        p = MIN_Prob
        # todo, how to get the confusion matrix of CNN recognition model

        if self.simulation == True:
            activity, _, _ = self.get_activity_from_dataset_by_time(self.cur_time)
            if activity == act:
                p = YOLO_ACC_OBJECT
            else:
                p = (1-YOLO_ACC_OBJECT)/(TOTAL_ACTIVITY_CNT-1) # totally 15 activities

        else:
            # find the probability of location from CNN recognition results
            p = self.object_prob
            pass

        # print('prob_of_object_using_vision object, act, p:', object, ' ', act, ' ', p)
        return p

    def prob_of_activity_in_dataset(self, act):
        "Get from dataset"
        p = PROB_OF_ALL_ACTIVITIES[act]

        return p



    def get_activity_from_dataset_by_time(self, time_str):

        # base_date = MILAN_BASE_DATE
        # day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days=i + 56)
        # day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

        # activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
        #     tools_ascc.get_activity_date(day_time_str)

        # # get expected_activity at current time (running time)
        # run_time = self.running_time.strftime(DATE_HOUR_TIME_FORMAT)

        test_time_str = '2009-12-11 12:58:33'
        # time_str = test_time_str
        expected_activity_str, expected_beginning_activity, expected_end_activity = \
            tools_ascc.get_activity(self.activity_date_dict, self.activity_begin_list,
                                    self.activity_end_list, time_str)
                
        return expected_activity_str, expected_beginning_activity, expected_end_activity
