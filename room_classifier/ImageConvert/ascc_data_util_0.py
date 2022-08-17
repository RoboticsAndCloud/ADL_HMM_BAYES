"""
@Brief: Process the data collected from ASCC environment.
@Author: Fei L
@Data: 03/14/2022
"""

from datetime import datetime
from datetime import timedelta

import os

DAY_FORMAT_STR = '%Y-%m-%d'
DATE_HOUR_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

DATE_HOUR_TIME_FORMAT_DIR = '%Y-%m-%d-%H-%M-%S'


ASCC_DATE_HOUR_TIME_FORMAT = '%Y%m%d%H%M%S'


TRAIN_RUNNING_TIME_FORMAT = "%H:%M:%S"


def get_timestamp_map_from_dir(dir, time_difference, copy_flag=False):
    path = dir
    count = 0
    prefix = ''
    map_dict = {}
    for fn in os.listdir(path):
        if os.path.isdir(dir + '/' + fn):
            print(fn)

            ascc_date_str = fn
            ascc_date = ascc_date_str.replace('_rotate', '')
            ascc_activity_time = datetime.strptime(ascc_date, ASCC_DATE_HOUR_TIME_FORMAT)

            ascc_activity_timestamp = ascc_activity_time.timestamp()
            milan_activity_timestamp = ascc_activity_timestamp - time_difference
            milan_activity_time = datetime.fromtimestamp(milan_activity_timestamp)

            mapping_time_str =  milan_activity_time.strftime(DATE_HOUR_TIME_FORMAT)
            print('milan activity mapping time:', mapping_time_str)

            mapping_time_str_dir =  milan_activity_time.strftime(DATE_HOUR_TIME_FORMAT_DIR)


            map_dict[ascc_date] = mapping_time_str

            source_dir = dir + '/' + fn

            new_dir = dir + '/' + mapping_time_str_dir

            print("new_dir:", new_dir)

            cp_cmd = 'cp -r ' + source_dir + ' ' + new_dir

            print(cp_cmd)
            
            try:
                if copy_flag:
                    os.system(cp_cmd)
            except Exception as e:
                print(e)
                pass

            # rotate_dir = source_dir + '_rotate'
            # print("rotate_dir:", rotate_dir)
            # if source_dir.find('rotate') > -1:
            #     continue
            
            # try:
            #     if os.path.exists(rotate_dir) == False:
            #         os.mkdir(rotate_dir)
            # except Exception as e:
            #     print(e)
            #     pass

            count = count +1

        else:
            print('fn:', fn)


    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return map_dict


def get_timestamp_map(test_dir, copy_flag = False):
    milan_activity_date = '2009-12-11 08:46:27'
    milan_activity_time = datetime.strptime(milan_activity_date, DATE_HOUR_TIME_FORMAT)
    milan_activity_time_start = milan_activity_time - timedelta(seconds = 60)

    ascc_date_str = '20220309213040_rotate'
    ascc_date = ascc_date_str.replace('_rotate', '')
    ascc_activity_time = datetime.strptime(ascc_date, ASCC_DATE_HOUR_TIME_FORMAT)

    milan_activity_time_start_time_stamp = milan_activity_time_start.timestamp()
    ascc_activity_timestamp = ascc_activity_time.timestamp()
    diff = ascc_activity_timestamp - milan_activity_time_start_time_stamp
    print('milan timestamp:', milan_activity_time_start_time_stamp, 'ascc:', ascc_activity_timestamp,'diff:', diff)

    # test_dir = '/home/ascc/Desktop/white_case_0309_1211/activity_data/kitchen_activity'
    map_dict = get_timestamp_map_from_dir(test_dir, diff, copy_flag)

    sd = sorted(map_dict.items())
    print('Mapping')
    print('Milan \t ASCC')
    for k,v in sd:
        print(v, '\t', k)

    for k,v in sd:
        print("'"+str(k)+"'" + ",")

    print('cnt:', len(map_dict))

def test_get_timestamp_map():
    milan_activity_date = '2009-12-11 08:46:27'
    milan_activity_time = datetime.strptime(milan_activity_date, DATE_HOUR_TIME_FORMAT)
    milan_activity_time_start = milan_activity_time - timedelta(seconds = 60)

    ascc_date_str = '20220309213040_rotate'
    ascc_date = ascc_date_str.replace('_rotate', '')
    ascc_activity_time = datetime.strptime(ascc_date, ASCC_DATE_HOUR_TIME_FORMAT)

    milan_activity_time_start_time_stamp = milan_activity_time_start.timestamp()
    ascc_activity_timestamp = ascc_activity_time.timestamp()
    diff = ascc_activity_timestamp - milan_activity_time_start_time_stamp
    print('milan timestamp:', milan_activity_time_start_time_stamp, 'ascc:', ascc_activity_timestamp,'diff:', diff)


    return ''

def generate_data_from_dir_kitchen(dir):
    path = dir
    count = 0
    source_dir_list = ['20220309214106',
'20220309214115',
'20220309214124',
'20220309214134',
'20220309214144',
'20220309214153',
'20220309214202',
'20220309214212',
'20220309214230',
'20220309214239',
'20220309214249',
'20220309214258',
'20220309214307',
'20220309214318',
'20220309214328',
'20220309214337',
'20220309214346',
'20220309214405',
'20220309214414',
'20220309214443',
'20220309214501',
'20220309214511',
'20220309214520',
'20220309214551',
'20220309214601',
'20220309214610',
'20220309214114',
'20220309214123',
'20220309214133',
'20220309214143',
'20220309214152',
'20220309214201',
'20220309214211',
'20220309214220',
'20220309214229',
'20220309214238',
'20220309214248',
'20220309214257',
'20220309214306',
'20220309214317',
'20220309214327',
'20220309214336',
'20220309214345',
'20220309214354',
'20220309214403',
'20220309214413',
'20220309214422',
'20220309214432',
'20220309214441',
'20220309214451',
'20220309214500',
'20220309214510',
'20220309214519',
'20220309214528',
'20220309214538',
'20220309214550',
'20220309214600',
'20220309214609',]

    map_dict = {}
    for fn in os.listdir(path):
        if os.path.isdir(dir + '/' + fn):
            print(fn)

            ascc_date_str = fn
            ascc_date = ascc_date_str.replace('_rotate', '')
            if ascc_date not in source_dir_list:
                continue

            ascc_activity_time = datetime.strptime(ascc_date, ASCC_DATE_HOUR_TIME_FORMAT)

            new_ascc_activity_time = ascc_activity_time + timedelta(seconds = 60 *15)

            new_ascc_time_str =  new_ascc_activity_time.strftime(ASCC_DATE_HOUR_TIME_FORMAT)

            print('new_ascc_time_str:', new_ascc_time_str)
            map_dict[ascc_date] = new_ascc_time_str

            source_dir = dir + '/' + fn

            new_dir = dir + '/' + new_ascc_time_str

            print("new_dir:", new_dir)

            mv_cmd = 'mv ' + source_dir + ' ' + new_dir

            print(mv_cmd)
            
            try:
                os.system(mv_cmd)
            except Exception as e:
                print(e)
                pass

            count = count +1

        else:
            print('fn:', fn)

    return map_dict


def generate_data_from_dir_kitchen_copy(dir):
    path = dir
    count = 0
    source_dir_list = ['20220309213605',
'20220309213615',
'20220309213624',
'20220309213634',
'20220309213643',
'20220309213652',
'20220309213702',
'20220309213712',
'20220309213721',
'20220309213730',
'20220309213740',
'20220309213749',
'20220309213759',
'20220309213808',
'20220309213817',
'20220309213827',
'20220309213836',
'20220309213846',
'20220309213855',
'20220309213904',
'20220309213914',
'20220309213923',
'20220309213932',
'20220309213942',
'20220309213951',
'20220309214001',
'20220309214010',
'20220309214019',
'20220309214029',
'20220309214038',
'20220309214047',
'20220309214056',
'20220309213614',
'20220309213623',
'20220309213633',
'20220309213642',
'20220309213651',
'20220309213701',
'20220309213711',
'20220309213720',
'20220309213729',
'20220309213739',
'20220309213748',
'20220309213758',
'20220309213807',
'20220309213816',
'20220309213826',
'20220309213835',
'20220309213845',
'20220309213854',
'20220309213903',
'20220309213913',
'20220309213922',
'20220309213931',
'20220309213941',
'20220309213950',
'20220309213959',
'20220309214009',
'20220309214018',
'20220309214028',
'20220309214037',
'20220309214046',
'20220309214055'
]

    map_dict = {}
    for fn in os.listdir(path):
        if os.path.isdir(dir + '/' + fn):
            print(fn)

            ascc_date_str = fn
            ascc_date = ascc_date_str.replace('_rotate', '')
            if ascc_date not in source_dir_list:
                continue

            ascc_activity_time = datetime.strptime(ascc_date, ASCC_DATE_HOUR_TIME_FORMAT)

            new_ascc_activity_time = ascc_activity_time + timedelta(seconds = 60 *15)

            new_ascc_time_str =  new_ascc_activity_time.strftime(ASCC_DATE_HOUR_TIME_FORMAT)

            print('new_ascc_time_str:', new_ascc_time_str)
            map_dict[ascc_date] = new_ascc_time_str



            source_dir = dir + '/' + fn

            new_dir = dir + '/' + new_ascc_time_str

            print("new_dir:", new_dir)

            cp_cmd = 'cp -r ' + source_dir + ' ' + new_dir

            print(cp_cmd)
            
            try:
                os.system(cp_cmd)
            except Exception as e:
                print(e)
                pass

            count = count +1

        else:
            print('fn:', fn)

    return map_dict
    

# get_timestamp_map()

# kitchen_dir = '/home/ascc/Desktop/white_case_0309_1211/activity_data/kitchen_activity'
# res = generate_data_from_dir_kitchen(kitchen_dir)
# sd = sorted(res.items())
# print('Mapping')
# print('Old \t New')
# for k,v in sd:
#     print(k, '\t', v)


# kitchen_dir = '/home/ascc/Desktop/white_case_0309_1211/activity_data/kitchen_activity'
# res = generate_data_from_dir_kitchen_copy(kitchen_dir)
# sd = sorted(res.items())
# print('Mapping')
# print('Old \t New')
# for k,v in sd:
#     print(k, '\t', v)

# Generate final data dir
# get_timestamp_map(copy_flag=True)



audio = '/home/ascc/Desktop/white_case_0309_1211/activity_data/kitchen_activity_data/Audio'

# get_timestamp_map(test_dir=audio)

# kitchen_dir = audio
# res = generate_data_from_dir_kitchen(kitchen_dir)
# sd = sorted(res.items())
# print('Mapping')
# print('Old \t New')
# for k,v in sd:
#     print(k, '\t', v)


# kitchen_dir = audio
# res = generate_data_from_dir_kitchen_copy(kitchen_dir)
# sd = sorted(res.items())
# print('Mapping')
# print('Old \t New')
# for k,v in sd:
#     print(k, '\t', v)

# Generate final data dir
# get_timestamp_map(test_dir=audio, copy_flag=True)




motion = '/home/ascc/Desktop/white_case_0309_1211/activity_data/kitchen_activity_data/Motion'

#get_timestamp_map(test_dir=motion)

# kitchen_dir = motion
# res = generate_data_from_dir_kitchen(kitchen_dir)
# sd = sorted(res.items())
# print('Mapping')
# print('Old \t New')
# for k,v in sd:
#     print(k, '\t', v)


# kitchen_dir = motion
# res = generate_data_from_dir_kitchen_copy(kitchen_dir)
# sd = sorted(res.items())
# print('Mapping')
# print('Old \t New')
# for k,v in sd:
#     print(k, '\t', v)

# Generate final data dir
# get_timestamp_map(test_dir=motion, copy_flag=True)