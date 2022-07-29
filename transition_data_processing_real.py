"""
@Brief: Analysis the motion data
@Source: https://docs.google.com/spreadsheets/d/1oWxLWxc0MSYiwQjNPDnw2ONWpzN88aEedCuV1nKmc0s/edit#gid=0
@Date: 03/13/2022

"""
from genericpath import exists
import math
from statistics import variance

"""
Type: walk
Mag avg: 2.3442230582482235
Mag max: 3.6444418151032454
Mag min: 0.31627270209646613
y avg: -9.206855799783984
y max: -6.943108199999999
y min: -12.6309652
Write into result: /home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170403/motion_mag.txt
Type: stand
Mag avg: 2.625329174570088
Mag max: 3.0667645199707785
Mag min: 2.217861048245954
y avg: -9.200192980674021
y max: -8.8652116
y min: -9.492837199999999
Write into result: /home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170036/motion_mag.txt
Type: sit on chair
Mag avg: 1.9579753596287521
Mag max: 2.4017253096013627
Mag min: 1.6418621027121738
y avg: -9.25221748419939
y max: -8.8652116
y min: -9.6889702
Write into result: /home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170003/motion_mag.txt
Type: sit in sofa
Mag avg: 1.8416399581282623
Mag max: 2.1881763884074887
Mag min: 1.4355154888219492
y avg: -9.285289001267204
y max: -8.9828914
y min: -9.610517
"""
DATA_SIZE_ONE_SECOND = 100 * 5 *2

SIT_MAX = 2.483
SIT_MIN = 1.435
SIT_AVG = 1.958

STAND_MAX = 3.006
STAND_MIN = 2.217
STAND_AVG = 2.625

SIT_MIN_Y = -9.688
SIT_MAX_Y = -8.865
SIT_AVG_Y = -9.285

WALK_AVG = 2.344

MOTION_SIT = 'sit'
MOTION_STAND = 'stand'
MOTION_WALK = 'walk'
MOTION_SQUAT = 'squat'

ASCC_DATA_SET_DIR = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/'

# Python program to get average of a list
def average(lst):
    return sum(lst) / len(lst)



def sliding_window(elements, window_size, x_list, y_list, z_list):
    if len(elements) <= window_size:
        return elements
    #print('winsize:', window_size)
    i = 0
    res = []
    while (i < len(elements)):
        w_list = elements[i:i+window_size]
        tmp_mag = average(w_list)

        w_xlist = x_list[i:i+window_size]
        w_ylist = y_list[i:i+window_size]
        w_zlist = z_list[i:i+window_size]

        x = average(w_xlist)
        y = average(w_ylist)
        z = average(w_zlist)
        var_x = variance(w_xlist)

        if tmp_mag <= SIT_MAX and tmp_mag >= SIT_MIN and var_x < 0.011:
            print('Sit detected, mag:', tmp_mag)
            res.append(MOTION_SIT)
        elif tmp_mag <= STAND_MAX and tmp_mag >= STAND_MIN  and var_x < 0.4:
            print('Stand detected, mag:', tmp_mag)
            res.append(MOTION_STAND)
        else:
            print('Walk detected, mag:', tmp_mag)
            res.append(MOTION_WALK)

        print('Mag avg:', average(w_list))
        print('Mag max:', max(w_list))
        print('Mag min:', min(w_list))

        print('x avg:', average(w_xlist))
        print('x max:', max(w_xlist))
        print('x min:', min(w_xlist))
        print('x var:', variance(w_xlist))

        print('y avg:', average(w_ylist))
        print('y max:', max(w_ylist))
        print('y min:', min(w_ylist))
        print('y var:', variance(w_ylist))

        print('z avg:', average(w_zlist))
        print('z max:', max(w_zlist))
        print('z min:', min(w_zlist))
        print('z var:', variance(w_zlist))
        # print('i:', i)
        i = i + window_size
        if i + window_size > len(elements):
            break

    return res


def sliding_window3(elements, window_size, z):
    if len(elements) <= window_size:
        return elements
    #print('winsize:', window_size)
    i = 0
    res = []
    while (i < len(elements)):
        w_list = elements[i:i+window_size]
        tmp_mag = average(w_list)

        if tmp_mag <= SIT_MAX and tmp_mag >= SIT_MIN and z < 2 and z > 1:
            print('Sit detected, mag:', tmp_mag)
            res.append(MOTION_SIT)
        elif tmp_mag <= STAND_MAX and tmp_mag >= STAND_MIN and z > 2 and z < 3:
            print('Stand detected, mag:', tmp_mag)
            res.append(MOTION_STAND)
        else:
            print('Walk detected, mag:', tmp_mag)
            res.append(MOTION_WALK)
        i = i + window_size
        if i + window_size > len(elements):
            break
        #print('i:', i)
    return res


def sliding_window2(elements, window_size):
    if len(elements) <= window_size:
        return elements
    #print('winsize:', window_size)
    i = 0
    res = []
    while (i < len(elements)):
        w_list = elements[i:i+window_size]
        tmp_mag = average(w_list)

        if tmp_mag <= SIT_MAX and tmp_mag >= SIT_MIN:
            print('Sit detected, mag:', tmp_mag)
            res.append(MOTION_SIT)
        elif tmp_mag <= STAND_MAX and tmp_mag >= STAND_MIN:
            print('Stand detected, mag:', tmp_mag)
            res.append(MOTION_STAND)
        else:
            print('Walk detected, mag:', tmp_mag)
            res.append(MOTION_WALK)
        i = i + window_size
        if i + window_size > len(elements):
            break
        #print('i:', i)
    return res


def motion_check(file_name, t = ''):
    mag_list = []
    mag_only_list = []

    x_list = []
    y_list = []
    z_list = []

    if not exists(file_name):
        return []

    with open(file_name, encoding = 'utf-8') as f:
        for line in f:
            m_d = line.split('\t')
            x = float(m_d[0])
            y = float(m_d[1])
            z = float(m_d[2])
            
            mag = 0
            mag = math.sqrt(x ** 2 + (abs(y)-9.81) ** 2 + z ** 2)
            #print("Accelerometer: x:", x, " y:", y, " z:", z, " Mag:", mag)
            mag_list.append((x, y, z, mag))
            mag_only_list.append(mag)

            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

    res = sliding_window(mag_only_list, int(DATA_SIZE_ONE_SECOND/2), x_list, y_list, z_list)
    # res = sliding_window2(mag_only_list, int(DATA_SIZE_ONE_SECOND/2)) # not accurate

    
    return res


def motion_process(file_name, t = ''):
    mag_list = []
    mag_only_list = []

    output = file_name.replace('.txt', '_mag.txt')
    print('output:', output)
    #return
    with open(file_name, encoding = 'utf-8') as f:
        for line in f:
            print(line, end = '')
            m_d = line.split('\t')
            x = float(m_d[0])
            y = float(m_d[1])
            z = float(m_d[2])
            
            mag = 0
            mag = math.sqrt(x ** 2 + (abs(y)-9.81) ** 2 + z ** 2)
            print("Accelerometer: x:", x, " y:", y, " z:", z, " Mag:", mag)
            mag_list.append((x, y, z, mag))
            mag_only_list.append(mag)

    print('Type:', t)
    print('Mag avg:', average(mag_only_list))
    print('Mag max:', max(mag_only_list))
    print('Mag min:', min(mag_only_list))

            
    with open(output, 'w') as f:
        for data in mag_list:
            x = data[0]
            y = data[1]
            z = data[2]
            mag = data[3]
            
            str_line = str(x) + '\t' + str(y) + '\t' + str(z) + '\t' + str(mag) + '\n'
            f.write(str_line)
            
    print('Write into result:', output)

#walk = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170403/motion.txt'
#motion_process(walk, 'walk')
#motion_check(walk, 'walk')

#stand = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170036/motion.txt'
#motion_process(stand, 'stand')
#motion_check(stand, 'stand')

#sit = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170003/motion.txt'
#motion_process(sit, 'sit on chair')
#motion_check(sit, 'sit on chair')

#sit = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311165922/motion.txt'
#motion_process(sit, 'sit on bed')
#motion_check(sit, 'sit on bed')


#sit = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311161310/motion.txt'
#motion_process(sit, 'sit in sofa')
#motion_check(sit, 'sit in sofa')

#sit_stand_sit = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170652/motion.txt'
#motion_process(sit, 'sit stand sit')
#motion_check(sit_stand_sit, 'sit stand sit')

# stand_walk_stand = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170610/motion.txt'
#motion_process(sit, 'sit stand sit')
# motion_check(stand_walk_stand, 'stand walk stand')

if __name__ == "__main__":

    # base_dir = '/home/ascc/Desktop/ascc_activity_real_data_0309/Motion/'
    base_dir = ASCC_DATA_SET_DIR + '/Motion/'
    #md = '/home/ascc/Desktop/ascc_activity_real_data_0309/Motion/2009-12-11-22-47-05'
    d_time = '2009-12-11-12-58-24'
    motion = base_dir + '/' + d_time + '/' + 'motion.txt'

    sit = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170003/motion.txt'
    sit = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311161310/motion.txt'
    walk = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170403/motion.txt'
    stand = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170036/motion.txt'
    stand_walk_stand = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170610/motion.txt'

    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-22-39-48/motion.txt'
    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-22-40-45/motion.txt'
    #test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-22-40-55/motion.txt'

    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-09-10-26/motion.txt'
    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-09-10-36/motion.txt'
    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-09-11-20/motion.txt'

    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-13-22-31/motion.txt'
    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-13-57-16/motion.txt'

    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-13-22-27/motion.txt'
    # test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-13-22-31/motion.txt'
    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-13-22-36/motion.txt'

    # test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-22-38-24/motion.txt'
    # test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-22-38-33/motion.txt'
    # test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-22-38-42/motion.txt'



    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-11-07-08/motion.txt'
    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-11-07-17/motion.txt'

    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-14-30-16/motion.txt'

    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-14-35-13/motion.txt'
    test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-14-35-04/motion.txt'
    #test = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170652/motion.txt' # stand
    #test = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311165922/motion.txt' # ist
    #test = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170003/motion.txt'  # sit
    #test = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170036/motion.txt'  # stand
    test = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170610/motion.txt'  # stand walk stand
    #test = '/home/ascc/Desktop/white_case_0309_1211/transition_motion/motion_0311/20220311170403/motion.txt'  # walk 


    #test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-18-40-23/motion.txt'

    # test = '/home/ascc/LF_Workspace/ReinforcementLearning/ASCC_Energy_Consumption/ASCC-RL-Algorithms_New_Reward_Test_Part/ascc_activity_real_data_0309/Motion/2009-12-11-13-56-19/motion.txt'

    res = motion_check(test, 'stand walk stand')
    print('res:', res)


