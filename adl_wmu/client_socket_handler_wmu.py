import socket
import argparse
import struct

import ADL_HMM_BAYES.adl_wmu.wmu_type_constants as wmu_type_constants

STATE_HEARTBEAT = 0
STATE_TEMPERATURE = 1
STATE_AUDIO = 2
STATE_CAMERA = 3
STATE_STEP = 4
STATE_TEXT_2_SPEECH = 5
STATE_ACTIVITY_TRIGGER = 6
STATE_ACTIVITY_TRIGGER_IMAGE = 7
STATE_ACTIVITY_TRIGGER_AUDIO = 8

def temperature_handler(ipsend, port):
    PORT = port
    IP = ipsend
    print("ip:", IP, "port:", PORT)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP, PORT))

    values = (STATE_TEMPERATURE)
    packer = struct.Struct('I')
    packed_data = packer.pack(values)
    s.send(packed_data)

    temperature = Temperature()

    values = (temperature)
    packer = struct.Struct('f')
    packed_data = packer.pack(values)
    s.send(packed_data)
    s.close()


def socket_image_sending_handler(ipsend, port, cnt, current_time, file):
    # todo open the image, get the lenght, send the lenth, send the data

    PORT = port
    IP = ipsend
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP, PORT))

    values = (STATE_ACTIVITY_TRIGGER_IMAGE)
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

    values = (wmu_type_constants.STATE_MEDICATION_ACTIVITY_WMU_AUDIO)
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

