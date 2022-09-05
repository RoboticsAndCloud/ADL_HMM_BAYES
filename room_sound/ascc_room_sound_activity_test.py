# -*- coding: utf-8 -*-
"""
@author: Jason Zhang
@github: https://github.com/JasonZhang156/Sound-Recognition-Tutorial
"""

from keras.models import load_model
import tensorflow as tf
import esc10_input
import numpy as np
import os
import time
import logging
from timeit import default_timer as timer

labels = ['door_open_closed', 'eating', 'keyboard', 'pouring_water_into_glass', 'toothbrushing', 'vacuum', 'drinking', 'flush_toilet', 'microwave', 'quiet', 'tv_news', 'washing_hand']
# labels = ['door_open_closed', 'eating', 'keyboard', 'pouring_water_into_glass', 'vacuum', 'drinking', 'flush_toilet', 'quiet', 'tv_news', 'washing_hand'] # 10

ASCC_DATA_NOTICE_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/ascc_data/notice.txt'
ASCC_DATA_RES_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/ascc_data/recognition_result.txt'

MODEL = load_model('./cnn_logmel_foldone_second.h5')

def read_dir_name(file_name):
    with open(file_name, 'r') as f:
        dir_name = str(f.read().strip())
        f.close()
    print('dir_name:%s', dir_name)
    return dir_name

def write_res_into_file(file_name, res_list):
    with open(file_name, 'w') as f:
        f.write(str(res_list))
        f.write('\t')
        f.close()
    
    return True

def use_gpu():
    """Configuration for GPU"""
    from tensorflow.compat.v1.keras.backend import set_session
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    # config = tf.ConfigProto()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    set_session(tf.compat.v1.InteractiveSession(config=config))


def run_cnn_model(path, cur_time):
    """
    Test model using test set
    :param test_fold: test fold of 5-fold cross validation
    :param feat: which feature to use
    """
    model = MODEL

    # 读取测试数据
    # _, _, test_features, test_labels = esc10_input.get_data(test_fold, feat)
    t1 = time.time()
    t2= time.time()

    # pre_test_dir = ''
    # while(True):
    #     test_dir = read_dir_name(ASCC_DATA_NOTICE_FILE)
    #     # logging.info('pre_test_dir:%s', pre_test_dir)
    #     logging.info('got cur test_dir:%s', test_dir)

    #     if pre_test_dir == test_dir:
    #         time.sleep(0.4)
    #         continue

    #     if os.path.exists(test_dir) == False:
    #         print('test_dir not exist:', test_dir)
    #         continue

    #     pre_test_dir = test_dir

    start = timer()

    try:
        t2= time.time()
        # path = test_dir + '/' + 'recorded.wav'
        # path = './mic_data/output.wav'
        # path = '/home/ascc/Downloads/total_sound_downsampling/cutting_food/cutting_food@4_freesound-cutting_food__538305__sound-2425__chopping-cutting.wav'
        test_feature = esc10_input.get_single_data(path)
        test_feature = np.expand_dims(test_feature, axis=-1)
        mean = np.mean(test_feature)
        std = np.std(test_feature)
        test_feature = (test_feature - mean) / std
        test_feature = test_feature.reshape(1,64,138,1)
        #print("test_feature: ",test_feature.shape)
        # 导入训练好的模型
        result = model.predict(test_feature)[0]
        # print(result)
        pre = labels[np.argmax(result)]
        val = result[np.argmax(result)]
        # print('pre:', pre, ' val:', val)
        print(pre + '(' + str(val) + ')')
        
        res = pre + '(' + str(val) + ')'
    except:
        print("ERROR........")
        res = ''


    end = timer()
    print("Get_prediction time cost:", end-start)   

    write_res_into_file(ASCC_DATA_RES_FILE, res)     

    # data = {DATA_TYPE : DATA_TYPE_SOUND, DATA_FILE:ASCC_DATA_RES_FILE, DATA_CURRENT: cur_time }
    # url = adl_env_client_lib.BASE_URL_NOTICE_RECOGNITION_RES
    # adl_env_client_lib.notice_post_handler(url, data)
    # print('Post the sound reconitioin event', path)        

    return result


# if __name__ == '__main__':
#     use_gpu()  # 使用GPU

#     acc = CNN_test(1, 'logmel')

    # dict_acc = {}
    # print('### [Start] Test model for ESC10 dataset #####')
    # for fold in [1, 2, 3, 4, 5]:
    #     print("## Start test fold{} model #####".format(fold))
    #     acc = CNN_test(fold, 'mfcc')
    #     dict_acc['fold{}'.format(fold)] = acc
    #     print("## Finish test fold{} model #####".format(fold))
    # dict_acc['mean'] = np.mean(list(dict_acc.values()))
    # print(dict_acc)
    # print('### [Finish] Test model finished for ESC10 dataset #####')


import adl_env_client_lib
import asyncio
import signal
import socketio
import functools
import time


# Update the IP Address according the target server
IP_ADDRESS = 'http://127.0.0.1:5000'
# Update your group ID
GROUP_ID = 1

INTERVAL = 10

shutdown = False


DATA_FILE_RECEIVED_FROM_WMU_EVENT_NAME = 'DATA_FILE_RECEIVED_FROM_WMU'
DATA_RECOGNITION_FROM_WMU_EVENT_NAME = 'DATA_RECOGNITION_FROM_WMU'

DATA_RECOGNITION_FINAL_TO_ADL_EVENT_NAME = 'DATA_RECOGNITION_TO_ADL'

DATA_TYPE = 'type'
DATA_CURRENT = 'current_time'
DATA_FILE = 'file'
DATA_TYPE_IMAGE = 'image'
DATA_TYPE_SOUND = 'audio'
DATA_TYPE_MOTION = 'motion'



# For getting the score
sio = socketio.AsyncClient()

@sio.event
async def connect():
    print('connection established')

@sio.on(DATA_FILE_RECEIVED_FROM_WMU_EVENT_NAME)
async def on_message(data):
    print('Got new data:', data)
    try:
        if data[DATA_TYPE] != DATA_TYPE_SOUND:
            return
        
        cur_time = data[DATA_CURRENT]
        file = data[DATA_FILE]
        print('cur_time:', cur_time, 'file:', file)
        
        run_cnn_model(file, cur_time)

    except Exception as e:
        print('Got error:', e)
        return
        pass
    
    event_name = DATA_RECOGNITION_FROM_WMU_EVENT_NAME
    data = {DATA_TYPE : DATA_TYPE_SOUND, DATA_FILE:ASCC_DATA_RES_FILE, DATA_CURRENT: cur_time }
    await sio.emit(event_name, data)
    print('send recognition :', data)


# @sio.on(DATA_RECOGNITION_FINAL_TO_ADL_EVENT_NAME)
# async def on_message(data):
#     try:
#         if data['type'] == DATA_TYPE_IMAGE:
#             print('Get image:', data)
#     except:
#         pass
#     print('Got final recognition data:', data)


@sio.event
async def disconnect():
    print('disconnected from server')

def stop(signame, loop):
    global shutdown
    shutdown = True

    tasks = asyncio.all_tasks()
    for _task in tasks:
        _task.cancel()

async def run():
    cnt = 0
    global shutdown
    while not shutdown:
        print('.', end='', flush=True)

        try:
            await asyncio.sleep(INTERVAL)
            cnt = cnt + INTERVAL
            print('run: ', cnt)
            # event_name = DATA_RECOGNITION_FROM_WMU_EVENT_NAME
            # broadcasted_data = {'type': DATA_TYPE_IMAGE, 'file': 'image0'}
            # await sio.emit(event_name, broadcasted_data)
        except asyncio.CancelledError as e:
            pass
            #print('run', 'CancelledError', flush=True)

    await sio.disconnect()

async def main():
    await sio.connect(IP_ADDRESS)

    loop = asyncio.get_running_loop()

    for signame in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signame),
            functools.partial(stop, signame, loop))

    task = asyncio.create_task(run())
    try:
        await asyncio.gather(task)
    except asyncio.CancelledError as e:
        pass
        #print('main', 'cancelledError')

    print('main-END')


if __name__ == '__main__':
    asyncio.run(main())

# if __name__ == "__main__":
#     print('Test running:===========================================================\n')

#     log.init_log("./log/my_program")  # ./log/my_program.log./log/my_program.log.wf7
#     logging.info("Hello World!!!")
#     # test()
#     # test_dnn()
    # run()
    
