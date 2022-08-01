"""

Write motion info file /home/pi/Desktop/data/motion/20220729101522//motion.txt
Write motion info file /home/pi/Desktop/data/motion/20220729101732//motion.txt
Write motion info file /home/pi/Desktop/data/motion/20220729101937//motion.txt
Write motion info file /home/pi/Desktop/data/motion/20220729102128//motion.txt

motion.txt

-1.1375714      -9.022118       -0.6668522
-1.1375714      -9.022118       -0.6668522
-1.0591182      -9.022118       -0.6668522
-1.1375714      -8.9436648      -0.6668522


33,Jogging,49105962326000,-0.6946377,12.680544,0.50395286;
33,Jogging,49106062271000,5.012288,11.264028,0.95342433;
33,Jogging,49106112167000,4.903325,10.882658,-0.08172209;
33,Jogging,49106222305000,-0.61291564,18.496431,3.0237172;
33,Jogging,49106332290000,-1.1849703,12.108489,7.205164;


columns = ['user', 'activity', 'time', 'x', 'y', 'z']


Walking = df[df['activity']=='Walking'].head(3555).copy()
Jogging = df[df['activity']=='Jogging'].head(3555).copy()
Upstairs = df[df['activity']=='Upstairs'].head(3555).copy()
Downstairs = df[df['activity']=='Downstairs'].head(3555).copy()
Sitting = df[df['activity']=='Sitting'].head(3555).copy()
Standing = df[df['activity']=='Standing'].copy()


"""
# cat standing.txt | grep motion.txt
file_dict = {
    'Sitting': ['20220729101522', '20220729101732', '20220729101937', '20220729102128'],
    'Standing': ['20220729102533', '20220729102707', '20220729102820', '20220729103103'],
    'Walking': ['20220729104834', '20220729104948', '20220729105116', '20220729105254'],
    'Jogging': ['20220729111246', '20220729111356', '20220729111504', '20220729111614'],
    'Laying': ['20220729103607', '20220729103739', '20220729103926', '20220729104053'],
    'Squating': ['20220729105913', '20220729110029', '20220729110140', '20220729110823']

}


file_dict_test = {
    'Sitting': ['20220729102338'],
    'Standing': ['20220729103312'],
    'Walking': ['20220729105500'],
    'Jogging': ['20220729111901'],
    'Laying': ['20220729104633'],
    'Squating': ['20220729110944']
}



MOTION_FOLDER = '/home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES/room_motion_activity/motion/'
MOTION_FOLDER_TEST = '/home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES/room_motion_activity/motion/test/'

MOTION_TXT = 'motion.txt'
sit_path_1 = MOTION_FOLDER + '20220729101522' + '/' + 'motion.txt'

TARGET_FILE = './ascc_dataset/ascc_v1_raw.txt'

def read_dir_name(file_name):
    with open(file_name, 'r') as f:
        dir_name = str(f.read().strip())
        f.close()
    print('dir_name:%s', dir_name)
    return dir_name

def write_res_into_file(file_name, res_list):
    with open(file_name, 'a+') as f:
        for v in res_list:
            f.write(str(v))
            f.write('\n')
        f.close()

    print('write_res_into_file, len:', len(res_list))
    
    return True


def convert(act, time, motion_file, target, user='ascc'):
    t_str_list = []
    cur_time = time
    with open(motion_file, 'r') as f:
        for index, line in enumerate(f):
            # print("Line {}: {}".format(index, line.strip()))

            s_str = str(line.strip())
            xyz_arr = s_str.split('\t')
            
            x = xyz_arr[0]
            y = xyz_arr[1]
            z = xyz_arr[2]

            cur_time = cur_time + 1

            t_str = str(user) + ',' + str(act) + ',' + str(cur_time) + ',' + str(x) + ',' + str(y) + ',' + str(z)

            t_str_list.append(t_str)

        f.close()
    print('act:', act)
    print('time:', cur_time)
    print('motion_file:', motion_file)
    print('target:', target)
    print('user:', user)
    print('len(t_str_list:', len(t_str_list))
    
    write_res_into_file(target, t_str_list)

    return len(t_str_list)

def test():
    act = 'Sitting'
    time = '20220729101522'
    motion_file = sit_path_1
    target = TARGET_FILE
    user = 'ascc'

    convert(act,time, motion_file, target, user)

def run_convert():
    act = 'Sitting'
    time = 1659407940  # 12585782270000
    motion_file = sit_path_1
    target = TARGET_FILE
    user = 'ascc'

    for k in file_dict.keys():
        act = k
        files = file_dict[k]
        for d in files:
            
            motion_file = MOTION_FOLDER + d + '/' + MOTION_TXT
            len = convert(act,time, motion_file, target, user)
            time = time + len
            print('=====================================')

        # print(k)

    return 0

run_convert()

