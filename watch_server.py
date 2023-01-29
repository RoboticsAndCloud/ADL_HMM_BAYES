#!/usr/bin/env python
import socket
# from medicine_socket_handler import *
import time
import requests
import sys
import os
import random
# import mysql.connector as mc
#from SpeakResults import SpeakResults
import struct
import threading

from timeit import default_timer as timer
import adl_env_client_lib




import adl_type_constants, adl_utils

"""
gnome-terminal -x bash -c "export PYTHONPATH=/usr/local/lib/python3.7/dist-packages:$PYTHONPATH && source ~/szd-python3-env/bin/activate && rosrun voice_interface medicine_server_main_node.py"

"""


DATA_FILE_RECEIVED_FROM_WMU_EVENT_NAME = 'DATA_FILE_RECEIVED_FROM_WMU'
DATA_RECOGNITION_FROM_WMU_EVENT_NAME = 'DATA_RECOGNITION_FROM_WMU'

DATA_RECOGNITION_FINAL_TO_ADL_EVENT_NAME = 'DATA_RECOGNITION_TO_ADL'

DATA_TYPE = 'type'
DATA_CURRENT = 'current_time'
DATA_FILE = 'file'
DATA_TYPE_IMAGE = 'image'
DATA_TYPE_SOUND = 'audio'
DATA_TYPE_MOTION = 'motion'



STATE_HEARTBEAT = 0
STATE_TEMPERATURE = 1
STATE_AUDIO = 2
STATE_CAMERA = 3
STATE_STEP = 4
STATE_TEXT_2_SPEECH = 5
STATE_ACTIVITY_TRIGGER = 6
STATE_ACTIVITY_TRIGGER_IMAGE = 7
STATE_ACTIVITY_TRIGGER_AUDIO = 8

# IPRECEIVE = "192.168.1.126"
# IPSEND = "192.168.1.135"
# #IPSEND = "192.168.1.131"
# PORT = 59000 # robot server
MODULATION = 0

##GLOBAL VALUES
global db_host
db_host = "31.22.4.94" #localhost - 31.22.4.94:3306
global db_user
db_user = "inhome" #root - inhome
global db_password
db_password = "T9@Jl4w6cF0Z(d" #Mm****** - T9@Jl4w6cF0Z(d
global db_name
db_name = "inhome_test" #healthsystem - inhome_test

state =-1
heartbeat = 0
temperature = 0
steps = 0


#MOTION_PATH = adl_type_constants.MOTION_FILE_SAVED_FOLDER
#WAV_PATH = adl_type_constants.AUDIO_FILE_SAVED_FOLDER
#CAM_PATH = adl_type_constants.IMAGE_FILE_SAVED_FOLDER
MOTION_PATH = ""
WAV_PATH = ""
MOTION_PATH = "/home/ascc/LF_Workspace/Bayes_model/Watch/Data/Motion/"
WAV_PATH = "/home/ascc/LF_Workspace/Bayes_model/Watch/Data/Audio/"
CAM_PATH = "/home/ascc/LF_Workspace/Bayes_model/Watch/Data/Image/"

IMAGE_CNT = 10
DATE_TIME_FORMAT = '%Y%m%d%H%M%S'


def util_mkdir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except Exception as e:
        print(e)
        return -1

    return 0


class MedicineServerMain(object):
    def __init__(self):
        pass

    def handler_service(self, conn):
        unpacker = struct.Struct('I')
        data = conn.recv(unpacker.size)
        print('size', unpacker.size)
        print('data:', data)
        data_int = int.from_bytes(data, "big")
        #unpacked_data = unpacker.unpack(data)
        #state = unpacked_data[0]
        state = data_int
        print('Got state:', state)

        if (state == "0"):
            print("heartbeat:", heartbeat)
            # function to send heartbeat to database
            # print(state)
            state = -1
            # print(state)

        elif (state == "5"):
            pass

        elif (state == adl_type_constants.STATE_ADL_ACTIVITY_WMU_IMAGE):
            cur_time, file = self.socket_image_handler(conn)

            # send notice to the server, and the server notice the CNN modle for recognition
            data = {DATA_TYPE : DATA_TYPE_IMAGE, DATA_FILE:file, DATA_CURRENT: cur_time }
            url = adl_env_client_lib.BASE_URL_NOTICE_FILES_RECEIVED
            #adl_env_client_lib.notice_post_handler(url, data)

            print('Post the image event', file)

        elif (state == adl_type_constants.STATE_ADL_ACTIVITY_WMU_AUDIO):
            cur_time, file = self.socket_audio_handler(conn)
            
            data = {DATA_TYPE : DATA_TYPE_SOUND, DATA_FILE:file, DATA_CURRENT: cur_time }

            url = adl_env_client_lib.BASE_URL_NOTICE_FILES_RECEIVED
            adl_env_client_lib.notice_post_handler(url, data)

            print('Post the audio event', file)


        elif (state == adl_type_constants.STATE_ADL_ACTIVITY_WMU_MOTION):
            cur_time, file = self.socket_motion_handler(conn)

            data = {DATA_TYPE : DATA_TYPE_MOTION, DATA_FILE:file, DATA_CURRENT: cur_time }
            # url = adl_env_client_lib.BASE_URL_PUSH
            # adl_env_client_lib.notice_request_handler(url)

            url = adl_env_client_lib.BASE_URL_NOTICE_FILES_RECEIVED
            adl_env_client_lib.notice_post_handler(url, data)

            print('Post the motion event', file)

        return 0

    def server(self):
        port = adl_type_constants.ROBOT_PORT
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        ip = adl_type_constants.ROBOT_IP
        ip = "10.227.102.0"
        s.bind((ip, port))
        print("Server established! IP:", ip)
        print("PORT:",port)
        s.listen(20)
        while True:
            conn, addr = s.accept()

            try:
                self.handler_service(conn)
            except Exception as e:
                print('error:', e)
                pass

            conn.close()

        return 0


    def socket_motion_handler(self, conn):
        # cnt, time
        # 1, 20220124101010
        unpacker = struct.Struct('14s')
        data = conn.recv(unpacker.size)
        unpacked_data = unpacker.unpack(data)
        data = unpacked_data
        current_time = data[0].decode()

        motion_folder = MOTION_PATH + current_time + '/'
        util_mkdir(motion_folder)

        filename = "motion"  # filename based on recording time
        mf = motion_folder + filename + '.txt'

        with open(mf, 'wb') as f:
            while True:
                l = conn.recv(1024)
                if not l: break
                f.write(l)
        print("Received motion:", mf)

        adl_utils.write_res_into_file(adl_type_constants.WMU_MOTION_FILE_NOTIFICATION_FILE, mf)


        return current_time, mf

    def socket_audio_handler(self, conn):
        # cnt, time
        # 1, 20220124101010
        unpacker = struct.Struct('14s')
        data = conn.recv(unpacker.size)
        unpacked_data = unpacker.unpack(data)
        data = unpacked_data
        current_time = data[0].decode()

        wav_folder = WAV_PATH + current_time + '/'
        util_mkdir(wav_folder)

        filename = "recorded"  # filename based on recording time
        wf = wav_folder + filename + '.wav'

        with open(wf, 'wb') as f:
            while True:
                l = conn.recv(1024)
                if not l: break
                f.write(l)
        print("Received Audio:", wf)

        adl_utils.write_res_into_file(adl_type_constants.WMU_AUDIO_FILE_NOTIFICATION_FILE, wf)


        return current_time, wf


    def socket_image_handler(self, conn):
        # cnt, time
        # 1, 20220124101010
        unpacker = struct.Struct('I 14s')
        data = conn.recv(unpacker.size)
        print('in image hanlder:', unpacker.size)
        print('in image hanlder:', data)
        cnt_byte = data[0:4]
        print('cnt_byte:', cnt_byte)
        unpacked_data = unpacker.unpack(data)
        data = unpacked_data

        #cnt = data[0]
        cnt = int.from_bytes(cnt_byte, "big")
        current_time = data[1].decode()

        image_dir = CAM_PATH + current_time + '/'
        util_mkdir(image_dir)

        image_file = image_dir + 'image' + str(cnt) + '.jpg'
        with open(image_file, 'wb') as f:
            while True:
                l = conn.recv(1024)
                if not l: break
                f.write(l)
        print("Received Image:", image_file)

        adl_utils.write_res_into_file(adl_type_constants.WMU_IMAGE_FILE_NOTIFICATION_FILE, image_file)


        return current_time, image_file


    def socket_images_audio_hander(self, conn):
        # now = datetime.now()
        # dt_string = now.strftime(DATE_TIME_FORMAT)
        # print("Date and time =", dt_string)
        # current_time = dt_string

        # image count
        cnt = 0
        while (cnt < IMAGE_CNT):
            res = self.socket_image_handler(conn)

            if res == -1:
                break
            cnt = cnt + 1
            # print(image_file)

        # audio
        self.socket_audio_handler(conn)

def server_run():
    server_node = MedicineServerMain()
    server_node.server()

    return 0

if __name__ == "__main__":


    server_run()

    # server = threading.Thread(target=server_run)
    # # web_server = threading.Thread(target=web_server_run)

    # server.start()
    # # web_server.start()

    # # server.join()
    # # web_server.join() # in seconds
    # # 
    # print("web start")
    # socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    


