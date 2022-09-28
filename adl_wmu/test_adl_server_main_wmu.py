import socket
# from service_socket_handler_wmu import *
import time
import requests
import sys
import os
import random
# import mysql.connector as mc
#from SpeakResults import SpeakResults
from datetime import datetime



import struct
import wmu_type_constants

STATE_HEARTBEAT = 0
STATE_TEMPERATURE = 1
STATE_AUDIO = 2
STATE_CAMERA = 3
STATE_STEP = 4
STATE_TEXT_2_SPEECH = 5
STATE_ACTIVITY_TRIGGER = 6
STATE_ACTIVITY_TRIGGER_IMAGE = 7
STATE_ACTIVITY_TRIGGER_AUDIO = 8

IPRECEIVE = "10.227.59.227"
IPSEND = "10.227.14.104"
#IPRECEIVE = "192.168.1.135"
#IPSEND = "192.168.1.128"
RECEIVE_PORT = 59100 # WMU server
SEND_PORT = 59000 # robot server

connect_limit = 20

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

DATE_TIME_FORMAT = '%Y%m%d%H%M%S'




def handler_service(conn):

    unpacker = struct.Struct('I')
    data = conn.recv(unpacker.size)
    unpacked_data = unpacker.unpack(data)
    state = unpacked_data[0]
    print('Got state:', state)

    if (state == "0"):
        heartbeat = SocketReceiveVariable(IPRECEIVE, PORT, MODULATION)
        MODULATION = MODULATION + 1
        print("heartbeat:", heartbeat)
        # function to send heartbeat to database
        # print(state)
        state = -1
        # print(state)

    elif (state == STATE_TEMPERATURE):
        temperature = temperature_handler(conn)
        print("Temperature:", temperature)
        state = -1
        # function to send temperature to database
    elif (state == "2"):
        SocketReceiveFile(IPRECEIVE, PORT, MODULATION)
        MODULATION = MODULATION + 1
        time.sleep(.5)
        SocketSendFile(IPSEND, PORT, MODULATION, 'TextToSpeech.wav')
        MODULATION = MODULATION + 1
        time.sleep(.5)
        state = -1
        # VOICESYSTEM
    elif (state == "3"):
        resp = requests.post('https://textbelt.com/text', {
            'phone': '5802284888',
            'message': 'A fall has been detected.',
            'key': '776f4f5c45ffe6f013ec59e0c4f6c8530d61f13bGgeRoNaXepXZdl5PPB1c5UWs3',
        })
        state = -1

    elif (state == "4"):
        steps = SocketReceiveVariable(IPRECEIVE, PORT, MODULATION)
        MODULATION = MODULATION + 1
        time.sleep(.5)
        print("Steps:", steps)
        state = -1

    elif (state == "5"):
        pass

    elif (state == STATE_ACTIVITY_TRIGGER_IMAGE):
        socket_image_handler(conn)
    elif (state == STATE_ACTIVITY_TRIGGER_AUDIO):
        socket_audio_handler(conn)
    elif (state == wmu_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_MOTION):
        socket_cmd_motion_handler(conn)
        #todo record audio and send the audio
    elif (state == wmu_type_constants.STATE_ENV_ACTIVITY_CMD_TAKING_FUSION):
        socket_cmd_fussion_handler(conn)
        # socket_cmd_taking_images_handler(conn)


    return 0



def socket_image_sending_handler(ipsend, port, cnt, current_time, file):
    # todo open the image, get the lenght, send the lenth, send the data

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ipsend, port))

    values = (wmu_type_constants.STATE_ADL_ACTIVITY_WMU_IMAGE)
    packer = struct.Struct('I')
    packed_data = packer.pack(values)
    s.send(packed_data)

    values = (cnt, current_time.encode())
    packer = struct.Struct('I 14s')
    packed_data = packer.pack(*values)
    s.send(packed_data)

    with open(file, 'rb') as f:
        for l in f: s.sendall(l)
    print('Image sent:', file)

    s.close()

    return ''



def socket_audio_sending_handler(ipsend, port, current_time, file):
    # todo open the image, get the lenght, send the lenth, send the data

    PORT = port
    IP = ipsend
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP, PORT))

    values = (wmu_type_constants.STATE_ADL_ACTIVITY_WMU_AUDIO)
    packer = struct.Struct('I')
    packed_data = packer.pack(values)
    s.send(packed_data)

    values = (current_time.encode())
    packer = struct.Struct('14s')
    packed_data = packer.pack(values)
    s.send(packed_data)

    with open(file, 'rb') as f:
        for l in f: s.sendall(l)
    print('Audio sent:', file)

    s.close()

    return 0

def socket_motion_sending_handler(ipsend, port, current_time, file):
    # todo open the image, get the lenght, send the lenth, send the data

    PORT = port
    IP = ipsend
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP, PORT))

    values = (wmu_type_constants.STATE_ADL_ACTIVITY_WMU_MOTION)
    packer = struct.Struct('I')
    packed_data = packer.pack(values)
    s.send(packed_data)

    values = (current_time.encode())
    packer = struct.Struct('14s')
    packed_data = packer.pack(values)
    s.send(packed_data)

    with open(file, 'rb') as f:
        for l in f: s.sendall(l)
    print('Motion sent:', file)

    s.close()

    return 0


def test_sending_motion(motion_file):
    now = datetime.now()
    dt_string = now.strftime(DATE_TIME_FORMAT)
    print("Date and time =", dt_string)

    current_time = dt_string
    
    try:
        socket_motion_sending_handler(wmu_type_constants.WMU_IPSEND, wmu_type_constants.WMU_SEND_PORT, current_time, motion_file)
    except Exception as e:
        print('send motion file error:', motion_file, ',', e)


def test_sending_image(image_file):
    now = datetime.now()
    dt_string = now.strftime(DATE_TIME_FORMAT)
    print("Date and time =", dt_string)

    current_time = dt_string
    cnt = 1
    
    try:
        socket_image_sending_handler(wmu_type_constants.WMU_IPSEND, wmu_type_constants.WMU_SEND_PORT, cnt, current_time, image_file)

    except Exception as e:
        print('send image file error:', image_file, ',', e)

def test_sending_audio(motion_file):
    now = datetime.now()
    dt_string = now.strftime(DATE_TIME_FORMAT)
    print("Date and time =", dt_string)

    current_time = dt_string
    
    try:
        socket_audio_sending_handler(wmu_type_constants.WMU_IPSEND, wmu_type_constants.WMU_SEND_PORT, current_time, motion_file)
    except Exception as e:
        print('send auido file error:', motion_file, ',', e)


def server():

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    print('default timeout:', socket.getdefaulttimeout())
    # socket.setdefaulttimeout(60)
    s.settimeout(0.2) # timeout for listening

    # ip = wmu_type_constants.WMU_IPRECEIVE
    # port = wmu_type_constants.WMU_RECEIVE_PORT
    # s.bind((ip, port))
    # print("Server established! IP:", ip)
    # print("PORT:", port)

    # s.listen(connect_limit)

    while True:
        # time out

        conn = None
        test_file = './test_sample/motion.txt'
        # test_file = './test_sample/motion_walking.txt'

        test_sending_motion(test_file)

        test_file = './test_sample/kitchen/image9_rotate.jpg'
        # test_file = './test_sample/read/image9_rotate.jpg'
        test_sending_image(test_file)

        test_file = './test_sample/quite.wav'
        test_sending_audio(test_file)


        time.sleep(10)


        continue

        try: 
            conn, addr = s.accept()
            print('conn:', conn)
        except socket.timeout:
            if conn is None:
                #print('conn is None, default motion')
                #motion_data_saver()
                continue
            pass
        except:
            raise
        else:
         # work with the connection, create a thread etc.
            handler_service(conn)
            conn.close()

    return 0


if __name__ == "__main__":
    server()

