"""
Brief: Bayes model for activity probability, including vision, motion, audio-based model.
Author: Frank
Date: 07/10/2022
"""


from multiprocessing import set_forkserver_preload


PROB_OF_ALL_ACTIVITIES = {'Bed_to_Toilet': 0.039303482587064675, 'Morning_Meds': 0.01791044776119403, 'Watch_TV': 0.05124378109452736, 'Kitchen_Activity': 0.2517412935323383, 'Chores': 0.011442786069651741, 'Leave_Home': 0.10248756218905472, 'Read': 0.14477611940298507, 'Guest_Bathroom': 0.15074626865671642, 'Master_Bathroom': 0.1328358208955224, 'Desk_Activity': 0.021890547263681594, 'Eve_Meds': 0.007462686567164179, 'Meditate': 0.00845771144278607, 'Dining_Rm_Activity': 0.009950248756218905, 'Master_Bedroom_Activity': 0.04975124378109453}

READ_ACTIVITY = 'read'
READINGROOM = 'readingroom'
BEDROOM = 'bedroom'
LIVINGROOM = 'livingroom'
OTHER = 'other'

# HMM Trans matrix, get it from motion_hmm.py
HMM_TRANS_MATRIX = {'Desk_Activity': {'Desk_Activity': 0.06451612903225806, 'Morning_Meds': 0.0, 'Leave_Home': 0.06451612903225806, 'Kitchen_Activity': 0.1935483870967742, 'Guest_Bathroom': 0.16129032258064516, 'Sleep': 0.0, 'Chores': 0.0967741935483871, 'Read': 0.0967741935483871, 'Master_Bathroom': 0.12903225806451613, 'Master_Bedroom_Activity': 0.0967741935483871, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.06451612903225806, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.03225806451612903}, 'Morning_Meds': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.2, 'Kitchen_Activity': 0.0, 'Guest_Bathroom': 0.24, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.48, 'Master_Bathroom': 0.0, 'Master_Bedroom_Activity': 0.04, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.04, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Leave_Home': {'Desk_Activity': 0.02702702702702703, 'Morning_Meds': 0.032432432432432434, 'Leave_Home': 0.17297297297297298, 'Kitchen_Activity': 0.21621621621621623, 'Guest_Bathroom': 0.10270270270270271, 'Sleep': 0.021621621621621623, 'Chores': 0.005405405405405406, 'Read': 0.2810810810810811, 'Master_Bathroom': 0.06486486486486487, 'Master_Bedroom_Activity': 0.032432432432432434, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.02702702702702703, 'Meditate': 0.016216216216216217, 'Dining_Rm_Activity': 0.0}, 'Kitchen_Activity': {'Desk_Activity': 0.010554089709762533, 'Morning_Meds': 0.026385224274406333, 'Leave_Home': 0.12928759894459102, 'Kitchen_Activity': 0.08443271767810026, 'Guest_Bathroom': 0.22163588390501318, 'Sleep': 0.005277044854881266, 'Chores': 0.0079155672823219, 'Read': 0.23746701846965698, 'Master_Bathroom': 0.12137203166226913, 'Master_Bedroom_Activity': 0.0395778364116095, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0158311345646438, 'Watch_TV': 0.08179419525065963, 'Meditate': 0.005277044854881266, 'Dining_Rm_Activity': 0.013192612137203167}, 'Guest_Bathroom': {'Desk_Activity': 0.01809954751131222, 'Morning_Meds': 0.0, 'Leave_Home': 0.12217194570135746, 'Kitchen_Activity': 0.3393665158371041, 'Guest_Bathroom': 0.08597285067873303, 'Sleep': 0.00904977375565611, 'Chores': 0.00904977375565611, 'Read': 0.1493212669683258, 'Master_Bathroom': 0.10407239819004525, 'Master_Bedroom_Activity': 0.10407239819004525, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.058823529411764705, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Sleep': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.06666666666666667, 'Kitchen_Activity': 0.26666666666666666, 'Guest_Bathroom': 0.0, 'Sleep': 0.0, 'Chores': 0.06666666666666667, 'Read': 0.0, 'Master_Bathroom': 0.5333333333333333, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.06666666666666667, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Chores': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.23809523809523808, 'Kitchen_Activity': 0.09523809523809523, 'Guest_Bathroom': 0.09523809523809523, 'Sleep': 0.047619047619047616, 'Chores': 0.0, 'Read': 0.19047619047619047, 'Master_Bathroom': 0.23809523809523808, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.0, 'Meditate': 0.09523809523809523, 'Dining_Rm_Activity': 0.0}, 'Read': {'Desk_Activity': 0.03319502074688797, 'Morning_Meds': 0.004149377593360996, 'Leave_Home': 0.07883817427385892, 'Kitchen_Activity': 0.46473029045643155, 'Guest_Bathroom': 0.1908713692946058, 'Sleep': 0.004149377593360996, 'Chores': 0.02074688796680498, 'Read': 0.04979253112033195, 'Master_Bathroom': 0.07053941908713693, 'Master_Bedroom_Activity': 0.024896265560165973, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.04149377593360996, 'Meditate': 0.004149377593360996, 'Dining_Rm_Activity': 0.012448132780082987}, 'Master_Bathroom': {'Desk_Activity': 0.02857142857142857, 'Morning_Meds': 0.02857142857142857, 'Leave_Home': 0.15, 'Kitchen_Activity': 0.3357142857142857, 'Guest_Bathroom': 0.05, 'Sleep': 0.014285714285714285, 'Chores': 0.04285714285714286, 'Read': 0.15, 'Master_Bathroom': 0.1, 'Master_Bedroom_Activity': 0.05, 'Bed_to_Toilet': 0.007142857142857143, 'Eve_Meds': 0.0, 'Watch_TV': 0.02142857142857143, 'Meditate': 0.014285714285714285, 'Dining_Rm_Activity': 0.007142857142857143}, 'Master_Bedroom_Activity': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.12698412698412698, 'Kitchen_Activity': 0.20634920634920634, 'Guest_Bathroom': 0.031746031746031744, 'Sleep': 0.031746031746031744, 'Chores': 0.0, 'Read': 0.19047619047619047, 'Master_Bathroom': 0.3968253968253968, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.015873015873015872, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Bed_to_Toilet': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.0, 'Kitchen_Activity': 0.0, 'Guest_Bathroom': 0.0, 'Sleep': 1.0, 'Chores': 0.0, 'Read': 0.0, 'Master_Bathroom': 0.0, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.0, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Eve_Meds': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.0, 'Kitchen_Activity': 0.2857142857142857, 'Guest_Bathroom': 0.42857142857142855, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.0, 'Master_Bathroom': 0.14285714285714285, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.14285714285714285, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}, 'Watch_TV': {'Desk_Activity': 0.04054054054054054, 'Morning_Meds': 0.0, 'Leave_Home': 0.05405405405405406, 'Kitchen_Activity': 0.40540540540540543, 'Guest_Bathroom': 0.3108108108108108, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.013513513513513514, 'Master_Bathroom': 0.08108108108108109, 'Master_Bedroom_Activity': 0.02702702702702703, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.013513513513513514, 'Watch_TV': 0.04054054054054054, 'Meditate': 0.013513513513513514, 'Dining_Rm_Activity': 0.0}, 'Meditate': {'Desk_Activity': 0.06666666666666667, 'Morning_Meds': 0.0, 'Leave_Home': 0.3333333333333333, 'Kitchen_Activity': 0.2, 'Guest_Bathroom': 0.0, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.06666666666666667, 'Master_Bathroom': 0.0, 'Master_Bedroom_Activity': 0.06666666666666667, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.06666666666666667, 'Meditate': 0.2, 'Dining_Rm_Activity': 0.0}, 'Dining_Rm_Activity': {'Desk_Activity': 0.0, 'Morning_Meds': 0.0, 'Leave_Home': 0.18181818181818182, 'Kitchen_Activity': 0.36363636363636365, 'Guest_Bathroom': 0.0, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.0, 'Master_Bathroom': 0.09090909090909091, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.0, 'Eve_Meds': 0.0, 'Watch_TV': 0.18181818181818182, 'Meditate': 0.09090909090909091, 'Dining_Rm_Activity': 0.09090909090909091}}
# HMM Start matrix
HMM_START_MATRIX = {'Desk_Activity': 0.0, 'Morning_Meds': 0.09090909090909091, 'Leave_Home': 0.11363636363636363, 'Kitchen_Activity': 0.25, 'Guest_Bathroom': 0.11363636363636363, 'Sleep': 0.0, 'Chores': 0.0, 'Read': 0.0, 'Master_Bathroom': 0.4090909090909091, 'Master_Bedroom_Activity': 0.0, 'Bed_to_Toilet': 0.022727272727272728, 'Eve_Meds': 0.0, 'Watch_TV': 0.0, 'Meditate': 0.0, 'Dining_Rm_Activity': 0.0}

# Read
# p1: location probability under one act
P1_Location_Under_Act = {
    READ_ACTIVITY:{READINGROOM: 0.9, BEDROOM: 0.05, LIVINGROOM:0.04, OTHER:0.01}
}


#prob_of_location_under_all_acts
Prob_Of_Location_Under_All_Act = {}


# for image recognition, we can get the reuslt for DNN, from the confusion matrix
DNN_ACC = 0.99
TOTAL_ACTIVITY_CNT = len(PROB_OF_ALL_ACTIVITIES)

MIN_Prob = 1e-20

# CNN Confusion matrix

class Bayes_Model_Vision(object):
    """
    This class is an implementation of the Bayes Model.
    """

    def __init__(self, act_name, location, simulation = False):
        self.act_name = act_name
        self.location = location
        self.simulation = simulation
        self.cur_time = ''

    def set_time(self, time):
        self.cur_time = time


    def get_prob(self, pre_activity, act_name, location):
        """ Return the state set of this model. """
        p = 0
        p =  self.prob_of_location_under_act(self.location, self.act_name) \
             * self.prob_prior_act(self.act_name) /(self.prob_of_location_under_all_acts(self.location))   * self.prob_of_location_vision(self.location)

        return p

    def prob_of_location_under_act(self, location, act_name):
        p = MIN_Prob

        try:
            p = P1_Location_Under_Act[act_name][location]
        except Exception as e:
            print('Got error from P_Location_Under_Act, location, act_name:', location, ', ', act_name, ', err:', e)

        return p

    def prob_prior_act(self, pre_activity, act_name):
        p = MIN_Prob
        try:
            p = HMM_TRANS_MATRIX[pre_activity][act_name]
        except Exception as e:
            print('Got error from HMM_TRANS_MATRIX, pre_activity, act_name:', pre_activity, ', ', act_name, ', err:', e)

        return p

    # Total probability rule, 15 activities
    def prob_of_location_under_all_acts(self, location):
        p = MIN_Prob

        for act in PROB_OF_ALL_ACTIVITIES.keys():
            p = p + self.prob_of_location_under_act(location, act) * self.prob_of_location_vision(act)

        return p
    
    # From CNN model, confusion matrix for simulation
    # For real time experiments, use CNN to predict the activity and get the probability
    def prob_of_location_vision(self, act):
        p = MIN_Prob
        # todo, how to get the confusion matrix of CNN recognition model

        if self.simulation == True:
            activity = get_activity_from_dataset_by_time(self.cur_time)
            if activity == act:
                p = DNN_ACC
            else:
                p = (1-DNN_ACC)/(TOTAL_ACTIVITY_CNT-1) # totally 15 activities

            pass

        else:
            pass

        return p
