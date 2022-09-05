import os
import struct
import socket


from picamera import PiCamera
from datetime import datetime
from datetime import timedelta
from timeit import default_timer as timer

import threading
import math
import time
import numpy as np

from gpiozero import LED
import board
import busio
import adafruit_adxl34x
import wave





import wmu_type_constants


WAV_PATH = "../data/audio/"
CAM_PATH = "../data/images/"
MO_PATH = "../data/motion/"

IMAGE_CNT = 5
AUDIO_DURATION = 2
MOTION_DURATION = 3

DATE_TIME_FORMAT = '%Y%m%d%H%M%S'
IMAGE_FLAG = 0

STATE_HEARTBEAT = 0
STATE_TEMPERATURE = 1
STATE_AUDIO = 2
STATE_CAMERA = 3
STATE_STEP = 4
STATE_TEXT_2_SPEECH = 5
STATE_ACTIVITY_TRIGGER_IMAGE = 6
STATE_ACTIVITY_TRIGGER_AUDIO = 7


import simpleaudio as sa
import pyaudio



g_picamera = PiCamera()
g_picamera.resolution = (640, 480)
g_picamera.rotation = 180




CHUNK = 44100  # frames to keep in buffer between reads
samp_rate = 44100  # sample rate [Hz]
pyaudio_format = pyaudio.paInt16  # 16-bit device
buffer_format = np.int16  # 16-bit for buffer
chans = 1  # only read 1 channel
dev_index = 1  # index of sound device

i2c = busio.I2C(board.SCL, board.SDA)
print(board.SCL, board.SDA)
accelerometer = adafruit_adxl34x.ADXL345(i2c)


print('DataRate.RATE_100_HZ:', adafruit_adxl34x.DataRate.RATE_100_HZ)
print('DataRate.RATE_200_HZ:', adafruit_adxl34x.DataRate.RATE_200_HZ)

print('Range.RANGE_2_G:', adafruit_adxl34x.Range.RANGE_2_G)

print('Rate:', accelerometer.data_rate)
print('Range:', accelerometer.range)
"""
DataRate.RATE_100_HZ: 10
DataRate.RATE_200_HZ: 11
Range.RANGE_2_G: 0
Rate: 10
Range: 0
"""

ACC_RATE = 100

state = -1
heart = 0
temperature = 0
activeTime = 1
blue = LED(24)
#button = Button(23)
step = 0



def speaker(file):
	wav_file = file

	try:
		w_object=sa.WaveObject.from_wave_file(wav_file)
		p_object = w_object.play()
		print("sound is playing")
		p_object.wait_done()
		print("finished")

	except FileNotFoundError:
		print("File not found")


def util_mkdir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except Exception as e:
        print(e)
        return -1

    return 0

def socket_audio_cmd_play_handler(conn):
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

    speaker(wf)

    return 0


def socket_image_handler(conn):
    # cnt, time
    # 1, 20220124101010
    unpacker = struct.Struct('I 14s')
    data = conn.recv(unpacker.size)
    unpacked_data = unpacker.unpack(data)
    data = unpacked_data

    print("socket image handlerdata:", data)
    cnt = data[0]
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

    return 0


def socket_images_audio_hander(conn):
    # now = datetime.now()
    # dt_string = now.strftime(DATE_TIME_FORMAT)
    # print("Date and time =", dt_string)
    # current_time = dt_string

    # image count
    cnt = 0
    while (cnt < IMAGE_CNT):
        res = socket_image_handler(conn)

        if res == -1:
            break

        cnt = cnt + 1
        # print(image_file)

    # audio
    socket_audio_handler(conn)


def temperature_handler(conn):
    unpacker = struct.Struct('f')
    data = conn.recv(unpacker.size)
    unpacked_data = unpacker.unpack(data)
    variable = unpacked_data[0]


    return variable


def pyserial_start():
    audio = pyaudio.PyAudio()  # create pyaudio instantiation
    ##############################
    ### create pyaudio stream  ###
    # -- streaming can be broken down as follows:
    # -- -- format             = bit depth of audio recording (16-bit is standard)
    # -- -- rate               = Sample Rate (44.1kHz, 48kHz, 96kHz)
    # -- -- channels           = channels to read (1-2, typically)
    # -- -- input_device_index = index of sound device
    # -- -- input              = True (let pyaudio know you want input)
    # -- -- frmaes_per_buffer  = chunk to grab and keep in buffer before reading
    ##############################
    stream = audio.open(format=pyaudio_format, rate=samp_rate, channels=chans, \
                        input_device_index=dev_index, input=True, \
                        frames_per_buffer=CHUNK)
    stream.stop_stream()  # stop stream to prevent overload
    return stream, audio


def pyserial_end():
    stream.close()  # close the stream
    audio.terminate()  # close the pyaudio connection


def data_grabber(rec_len, stream):
    stream.start_stream()  # start data stream
    stream.read(CHUNK, exception_on_overflow=False)  # flush port first
    # t_0 = datetime.datetime.now() # get datetime of recording start
    print('Recording Started.')
    data, data_frames = [], []  # variables
    for frame in range(0, int((samp_rate * rec_len) / CHUNK)):
        # grab data frames from buffer
        stream_data = stream.read(CHUNK, exception_on_overflow=False)
        data_frames.append(stream_data)  # append data
        data.append(np.frombuffer(stream_data, dtype=buffer_format))
    stream.stop_stream()  # stop data stream
    print('Recording Stopped.')
    return data, data_frames


def data_saver(wav_folder, data_frames, audio):
    data_folder = '/home/pi/Desktop/data'  # folder where data will be saved locally
    data_folder = wav_folder
    util_mkdir(wav_folder)
    # if os.path.isdir(data_folder)==False:
    #    os.mkdir(data_folder) # create folder if it doesn't exist
    filename = "recorded"  # filename based on recording time
    wf = wave.open(data_folder + filename + '.wav', 'wb')  # open .wav file for saving
    global chans
    global pyaudio_format
    global samp_rate
    wf.setnchannels(chans)  # set channels in .wav file
    wf.setsampwidth(audio.get_sample_size(pyaudio_format))  # set bit depth in .wav file
    wf.setframerate(samp_rate)  # set sample rate in .wav file
    wf.writeframes(b''.join(data_frames))  # write frames in .wav file
    wf.close()  # close .wav file
    return filename

def motion_data_saver():

    motion_file = MO_PATH

    now = datetime.now()
    dt_string = now.strftime(DATE_TIME_FORMAT)
    print("Date and time =", dt_string)

    current_time = dt_string
    mo_dir = motion_file + current_time + '/'
    util_mkdir(mo_dir)
    
    
    start_t = timer()
    
    motion_list = []
    while True:
        end_t = timer()
        duration = end_t - start_t
        
        # record 2 seconds motion data
        if duration > MOTION_DURATION:
       	    print('motion recorded duration:', duration)
            
            break
        
        
        (x, y, z) = accelerometer.acceleration
        motion_list.append((x, y, z))
        
        acc = 0
        acc = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        #acc = math.sqrt(x ** 2 + (abs(y)-9.81) ** 2 + (abs(z)) ** 2)
        print("Accelerometer: x:", x, " y:", y, " z:", z, " Mag:", acc)
        acc = round(acc, 4)
        # due to the accelerometer 100, need to sleep and wait
        time.sleep(1.0/(ACC_RATE + 20))
        
    print('Len motion data:', len(motion_list))
    
    # write date into file
    motion_file = mo_dir + '/' + 'motion.txt'
    with open(motion_file, 'w') as f:
        for data in motion_list:
            x = data[0]
            y = data[1]
            z = data[2]
            str_line = str(x) + '\t' + str(y) + '\t' + str(z) + '\n'
            f.write(str_line)
            
    print('Write motion info file', motion_file)

    try:
        socket_motion_sending_handler(wmu_type_constants.WMU_IPSEND, wmu_type_constants.WMU_SEND_PORT, current_time, motion_file)
    except Exception as e:
        print('send motion file error:', motion_file, ',', e)


def audio_capture(current_time):
    time.sleep(.1)
    stream, audio = pyserial_start()  # start the pyaudio stream
    record_length = AUDIO_DURATION  # seconds to record
    # input('Press Enter to Record Noise (Keep Quiet!)')
    global CHUNK
    # noise_chunks, _ = data_grabber(CHUNK / samp_rate, stream)  # grab the data
    blue.on()
    # input('Press Enter to Record Data (Turn Freq. Generator On)')
    data_chunks, data_frames = data_grabber(record_length, stream)  # grab the data
    blue.off()
    wav_folder = WAV_PATH + current_time + '/'
    data_saver(wav_folder, data_frames, audio)  # save the data as a .wav file

    # pyserial_end()  # close the stream/pyaudio connection
    stream.close()  # close the stream
    audio.terminate()  # close the pyaudio connection

    time.sleep(.1)
    file = wav_folder + "recorded.wav"

    # file = "Feedback.wav"
    # Speaker(file)

    return ''


def photo_audio_capture():
    # capture 10 images for the activities
    # resolution: 1920* 1080
    # capture 5 seconds audio, 431K
    # todo: it takes 5 seconds to take and save images, 1) reduce image size 2) do not save the image on client

    camera = g_picamera
    image_file = CAM_PATH

    now = datetime.now()
    dt_string = now.strftime(DATE_TIME_FORMAT)
    print("Date and time =", dt_string)

    current_time = dt_string
    image_dir = image_file + current_time + '/'
    util_mkdir(image_dir)

    cnt = 0

    # cam_stream = BytesIO()
    # cam_stream_list = []

    while (cnt < IMAGE_CNT):
        image_file = image_dir + 'image' + str(cnt) + '.jpg'
        camera.capture(image_file)

        # camera.capture(cam_stream, 'jpeg')
        # cam_stream_list.append(cam_stream)

        cnt = cnt + 1
        # print(image_file)

    start_t = timer()
    audio_capture(current_time)
    end_t = timer()
    print("Audio capture takes:", end_t - start_t)

    ### handler, sending the data
    cnt = 0
    start_t = timer()

    while (cnt < IMAGE_CNT):
        image_file = image_dir + 'image' + str(cnt) + '.jpg'

        try:
            socket_image_sending_handler(wmu_type_constants.WMU_IPSEND, wmu_type_constants.WMU_SEND_PORT, cnt, current_time, image_file)
        except Exception as e:
            print('send image file error', image_file)



        cnt = cnt + 1
        # print(image_file)

    wav_file = WAV_PATH + current_time + '/' + 'recorded.wav'
    try:
        socket_audio_sending_handler(wmu_type_constants.WMU_IPSEND, wmu_type_constants.WMU_SEND_PORT, current_time, wav_file)
    except Exception as e:
        print('Sending audio file error', wav_file)

    end_t = timer()
    print("Sending files takes:", end_t - start_t)

    return ''



def socket_cmd_fussion_handler(conn):
    t_photo_audio = threading.Thread(target=photo_audio_capture)
    t_motion = threading.Thread(target=motion_data_saver)

    t_photo_audio.start()
    t_motion.start()

    t_photo_audio.join(6)
    t_motion.join(7) # in seconds

    return 0

def socket_cmd_motion_handler(conn):
    motion_data_saver()
    return 0

def socket_cmd_taking_images_handler(conn):
    # capture 10 images for the activities
    # resolution: 1920* 1080
    # capture 5 seconds audio, 431K
    # todo: it takes 5 seconds to take and save images, 1) reduce image size 2) do not save the image on client

    camera = g_picamera
    image_file = CAM_PATH

    now = datetime.now()
    dt_string = now.strftime(DATE_TIME_FORMAT)
    print("Date and time =", dt_string)

    current_time = dt_string
    image_dir = image_file + current_time + '/'
    util_mkdir(image_dir)

    cnt = 0

    # cam_stream = BytesIO()
    # cam_stream_list = []

    while (cnt < IMAGE_CNT):
        image_file = image_dir + 'image' + str(cnt) + '.jpg'
        camera.capture(image_file)

        # camera.capture(cam_stream, 'jpeg')
        # cam_stream_list.append(cam_stream)

        cnt = cnt + 1
        # print(image_file)

    ### handler, sending the data
    cnt = 0
    start_t = timer()

    while (cnt < IMAGE_CNT):
        image_file = image_dir + 'image' + str(cnt) + '.jpg'

        socket_image_sending_handler(wmu_type_constants.WMU_IPSEND, wmu_type_constants.WMU_SEND_PORT, cnt, current_time, image_file)

        cnt = cnt + 1
        # print(image_file)

    end_t = timer()
    print("Sending files takes:", end_t - start_t)

    return ''

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

def test_sending(motion_file):
    now = datetime.now()
    dt_string = now.strftime(DATE_TIME_FORMAT)
    print("Date and time =", dt_string)

    current_time = dt_string
    
    try:
        socket_motion_sending_handler(wmu_type_constants.WMU_IPSEND, wmu_type_constants.WMU_SEND_PORT, current_time, motion_file)
    except Exception as e:
        print('send motion file error:', motion_file, ',', e)
