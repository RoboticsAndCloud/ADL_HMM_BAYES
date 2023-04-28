import socket
# from SocketReceive import *
# from SocketSend import *
# from SocketHandler import *
import time
# from picamera import PiCamera
import pyaudio
# from Speaker import Speaker
# from gpiozero import LED
import numpy as np
# import time, wave, datetime, os, csv
# from gpiozero import Button
# from Heartbeat import ReadHeartbeat
# from getHeartbeat import getHeartbeat
# from Temperature import Temperature
# import board
# import busio
# import adafruit_adxl34x
# import math

from datetime import datetime
from datetime import timedelta
from timeit import default_timer as timer

import os
import wave

import cv2 as cv
import re
import time  
from timeit import default_timer as timer
import threading

from io import BytesIO

import struct

STATE_HEARTBEAT = 0
STATE_TEMPERATURE = 1
STATE_AUDIO = 2
STATE_CAMERA = 3
STATE_STEP = 4
STATE_TEXT_2_SPEECH = 5
STATE_ACTIVITY_TRIGGER = 6
STATE_ACTIVITY_TRIGGER_IMAGE = 7
STATE_ACTIVITY_TRIGGER_AUDIO = 8

WAV_PATH = "/home/ascc/lf_workspace/IROS23/data/audio/"
CAM_PATH = "/home/ascc/lf_workspace/IROS23/data/image/"
MO_PATH = "/home/pi/Desktop/data/motion/"

IMAGE_CNT = 10
AUDIO_DURATION = 1

DATE_TIME_FORMAT = '%Y%m%d%H%M%S'

g_picamera = None
# g_picamera = PiCamera()
# g_picamera.resolution = (640, 480)
# camera.framerate = 15

G_PARAMETER = 9.81

IPSEND = "10.227.96.41"
IPRECEIVE = "192.168.1.131"
PORT = 49152
MODULATION = 0

CHUNK = 44100  # frames to keep in buffer between reads
samp_rate = 44100  # sample rate [Hz]
pyaudio_format = pyaudio.paInt16  # 16-bit device
buffer_format = np.int16  # 16-bit for buffer
chans = 1  # only read 1 channel
dev_index = 1  # index of sound device

# i2c = busio.I2C(board.SCL, board.SDA)
# print(board.SCL, board.SDA)
# accelerometer = adafruit_adxl34x.ADXL345(i2c)


# print('DataRate.RATE_100_HZ:', adafruit_adxl34x.DataRate.RATE_100_HZ)
# print('DataRate.RATE_200_HZ:', adafruit_adxl34x.DataRate.RATE_200_HZ)

# print('Range.RANGE_2_G:', adafruit_adxl34x.Range.RANGE_2_G)

# print('Rate:', accelerometer.data_rate)
# print('Range:', accelerometer.range)
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
# blue = LED(24)
# button = Button(23)
button = None
step = 0


source = 0
w = 299
h = 299

PHOTO_IMAGE_POST_FILE_NAME = '/home/ascc/asccbot_v3/asccChatBot/AsccChatbot/Images/photo_quickly_taking_post.config'
CAMERA_INIT_TIME = 2 # Seconds for initialing the camera or you may get a green photo

def getCurrentTimestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def store_name_to_file(file_name, name):
    with open(file_name, 'w') as f:
        f.write(name)
        f.close()
    return ''


def util_mkdir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except Exception as e:
        print(e)
        return -1

    return 0


def create_capture(source = 0, fallback = None):
    '''source: <int> or '<int>|<filename>|synth [:<param_name>=<value> [:...]]'
    '''
    global w,h 
    source = str(source).strip()

    # Win32: handle drive letter ('c:', ...)
    source = re.sub(r'(^|=)([a-zA-Z]):([/\\a-zA-Z0-9])', r'\1?disk\2?\3', source)
    chunks = source.split(':')
    chunks = [re.sub(r'\?disk([a-zA-Z])\?', r'\1:', s) for s in chunks]

    source = chunks[0]
    try: source = int(source)
    except ValueError: pass
    params = dict( s.split('=') for s in chunks[1:] )

    cap = None
    if source == 'synth':
        Class = classes.get(params.get('class', None), VideoSynthBase)
        try: cap = Class(**params)
        except: pass
    else:
        cap = cv.VideoCapture(source)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
        if 'size' in params:
            w, h = map(int, params['size'].split('x'))
            cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        if fallback is not None:
            return create_capture(fallback, None)
    return cap



def photo_capture(current_time):
    caps = list(map(create_capture, [0]))

    shot_idx = 0
    image_cnt = 3
    start_t = timer()


    cam_folder = CAM_PATH + current_time + '/'
    util_mkdir(cam_folder)

    while True:
        imgs = []
        for i, cap in enumerate(caps):
            ret, img = cap.read()
            imgs.append(img)
            # cv.imshow('capture %d' % i, img)
            # time.sleep(10)

        end_t = timer()
        duration = end_t - start_t
        if duration < CAMERA_INIT_TIME:
            continue

        # ch = cv.waitKey(1)
        if True:
        # if ch == ord(' '):
            for i, img in enumerate(imgs):
                fn = '%s/shot_%d_%03d.jpg' % (cam_folder, i, shot_idx)
                cv.imwrite(fn, img)
                store_name_to_file(PHOTO_IMAGE_POST_FILE_NAME, fn)
                print(fn, 'saved')
                time.sleep(0.1)
            shot_idx += 1
            if shot_idx == image_cnt:
                # cv.destroyAllWindows()
                break
            



# def photo_audio_capture():
#     # capture 10 images for the activities
#     # resolution: 1920* 1080
#     # capture 5 seconds audio, 431K
#     # todo: it takes 5 seconds to take and save images, 1) reduce image size 2) do not save the image on client

#     camera = g_picamera
#     image_file = CAM_PATH

#     now = datetime.now()
#     dt_string = now.strftime(DATE_TIME_FORMAT)
#     print("Date and time =", dt_string)

#     current_time = dt_string
#     image_dir = image_file + current_time + '/'
#     util_mkdir(image_dir)

#     cnt = 0

#     # cam_stream = BytesIO()
#     # cam_stream_list = []

#     while (cnt < IMAGE_CNT):
#         image_file = image_dir + 'image' + str(cnt) + '.jpg'
#         camera.capture(image_file)

#         # camera.capture(cam_stream, 'jpeg')
#         # cam_stream_list.append(cam_stream)

#         cnt = cnt + 1
#         # print(image_file)

#     start_t = timer()
#     audio_capture(current_time)
#     end_t = timer()
#     print("Audio capture takes:", end_t - start_t)

#     ### handler, sending the data
#     cnt = 0
#     start_t = timer()

#     while (cnt < IMAGE_CNT):
#         image_file = image_dir + 'image' + str(cnt) + '.jpg'

#         socket_image_sending_handler(IPSEND, PORT, cnt, current_time, image_file)

#         cnt = cnt + 1
#         # print(image_file)

#     wav_file = WAV_PATH + current_time + '/' + 'recorded.wav'
#     socket_audio_sending_handler(IPSEND, PORT, current_time, wav_file)
#     end_t = timer()
#     print("Sending files takes:", end_t - start_t)

#     return ''


# def photo_capture():
#     # capture 10 images for the activities
#     # resolution: 1920* 1080
#     # todo: it takes 5 seconds to take and save images, 1) reduce image size 2) do not save the image on client

#     camera = g_picamera
#     image_file = CAM_PATH

#     now = datetime.now()
#     dt_string = now.strftime(DATE_TIME_FORMAT)
#     print("Date and time =", dt_string)

#     current_time = dt_string
#     image_dir = image_file + current_time + '/'
#     util_mkdir(image_dir)

#     cnt = 0

#     while (cnt < IMAGE_CNT):
#         image_file = image_dir + 'image' + str(cnt) + '.jpg'
#         # camera.start_preview()
#         camera.capture(image_file)
#         # camera.stop_preview()

#         cnt = cnt + 1

#     # SocketSendVariable(IPSEND, PORT, MODULATION, "3")
#     return ''

def audio_capture(current_time):

    # the file name output you want to record into
    wav_folder = WAV_PATH + current_time + '/'
    util_mkdir(wav_folder)


    filename = wav_folder + "recorded.wav"
    # set the chunk size of 1024 samples
    chunk = 1024
    # sample format
    FORMAT = pyaudio.paInt16
    # mono, change to 2 if you want stereo
    channels = 1
    # 44100 samples per second
    sample_rate = 44100
    record_seconds = 2
    # initialize PyAudio object
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()
    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()
    

# def audio_capture2(current_time):
#     # SocketSendVariable(IPSEND, PORT, MODULATION, "2")
#     global MODULATION
#     MODULATION = MODULATION + 1
#     time.sleep(.1)
#     stream, audio = pyserial_start()  # start the pyaudio stream
#     record_length = AUDIO_DURATION  # seconds to record
#     # input('Press Enter to Record Noise (Keep Quiet!)')
#     global CHUNK
#     # noise_chunks, _ = data_grabber(CHUNK / samp_rate, stream)  # grab the data
#     blue.on()
#     # input('Press Enter to Record Data (Turn Freq. Generator On)')
#     data_chunks, data_frames = data_grabber(record_length, stream)  # grab the data
#     blue.off()
#     wav_folder = WAV_PATH + current_time + '/'
#     data_saver(wav_folder, data_frames, audio)  # save the data as a .wav file

#     # pyserial_end()  # close the stream/pyaudio connection
#     stream.close()  # close the stream
#     audio.terminate()  # close the pyaudio connection

#     time.sleep(.1)
#     file = wav_folder + "recorded.wav"

#     # file = "Feedback.wav"
#     # Speaker(file)

#     return ''


# def pyserial_start():
#     audio = pyaudio.PyAudio()  # create pyaudio instantiation
#     ##############################
#     ### create pyaudio stream  ###
#     # -- streaming can be broken down as follows:
#     # -- -- format             = bit depth of audio recording (16-bit is standard)
#     # -- -- rate               = Sample Rate (44.1kHz, 48kHz, 96kHz)
#     # -- -- channels           = channels to read (1-2, typically)
#     # -- -- input_device_index = index of sound device
#     # -- -- input              = True (let pyaudio know you want input)
#     # -- -- frmaes_per_buffer  = chunk to grab and keep in buffer before reading
#     ##############################
#     stream = audio.open(format=pyaudio_format, rate=samp_rate, channels=chans, \
#                         input_device_index=dev_index, input=True, \
#                         frames_per_buffer=CHUNK)
#     stream.stop_stream()  # stop stream to prevent overload
#     return stream, audio


# def pyserial_end():
#     stream.close()  # close the stream
#     audio.terminate()  # close the pyaudio connection


# def data_grabber(rec_len, stream):
#     stream.start_stream()  # start data stream
#     stream.read(CHUNK, exception_on_overflow=False)  # flush port first
#     # t_0 = datetime.datetime.now() # get datetime of recording start
#     print('Recording Started.')
#     data, data_frames = [], []  # variables
#     for frame in range(0, int((samp_rate * rec_len) / CHUNK)):
#         # grab data frames from buffer
#         stream_data = stream.read(CHUNK, exception_on_overflow=False)
#         data_frames.append(stream_data)  # append data
#         data.append(np.frombuffer(stream_data, dtype=buffer_format))
#     stream.stop_stream()  # stop data stream
#     print('Recording Stopped.')
#     return data, data_frames


# def data_saver(wav_folder, data_frames, audio):
#     data_folder = '/home/pi/Desktop/data'  # folder where data will be saved locally
#     data_folder = wav_folder
#     util_mkdir(wav_folder)
#     # if os.path.isdir(data_folder)==False:
#     #    os.mkdir(data_folder) # create folder if it doesn't exist
#     filename = "recorded"  # filename based on recording time
#     wf = wave.open(data_folder + filename + '.wav', 'wb')  # open .wav file for saving
#     global chans
#     global pyaudio_format
#     global samp_rate
#     wf.setnchannels(chans)  # set channels in .wav file
#     wf.setsampwidth(audio.get_sample_size(pyaudio_format))  # set bit depth in .wav file
#     wf.setframerate(samp_rate)  # set sample rate in .wav file
#     wf.writeframes(b''.join(data_frames))  # write frames in .wav file
#     wf.close()  # close .wav file
#     return filename

# def motion_data_saver():

#     motion_file = MO_PATH

#     now = datetime.now()
#     dt_string = now.strftime(DATE_TIME_FORMAT)
#     print("Date and time =", dt_string)

#     current_time = dt_string
#     mo_dir = motion_file + current_time + '/'
#     util_mkdir(mo_dir)
    
    
#     start_t = timer()
    
#     motion_list = []
#     while True:
#         end_t = timer()
#         duration = end_t - start_t
        
#         # record 1 seconds motion data
#         if duration > 10:
#        	    print('motion recorded duration:', duration)
#             break
        
#         (x, y, z) = accelerometer.acceleration
#         motion_list.append((x, y, z))
        
#         acc = 0
#         acc = math.sqrt(x ** 2 + y ** 2 + z ** 2)
#         #acc = math.sqrt(x ** 2 + (abs(y)-9.81) ** 2 + (abs(z)) ** 2)
#         print("Accelerometer: x:", x, " y:", y, " z:", z, " Mag:", acc)
#         acc = round(acc, 4)
#         # due to the accelerometer 100, need to sleep and wait
#         time.sleep(1.0/(ACC_RATE + 20))
        
#     print('Len motion data:', len(motion_list))
    
#     # write date into file
#     motion_file = mo_dir + '/' + 'motion.txt'
#     with open(motion_file, 'w') as f:
#         for data in motion_list:
#             x = data[0]
#             y = data[1]
#             z = data[2]
#             str_line = str(x) + '\t' + str(y) + '\t' + str(z) + '\n'
#             f.write(str_line)
            
#     print('Write motion info file', motion_file)


while (True):

    # default event
    if activeTime % 10 == 0: 
        pass
        #temperature_handler(IPSEND, PORT)

    if (state == 0):
        zero = ReadHeartbeat()
        heart = getHeartbeat()
        SocketSendVariable(IPSEND, PORT, MODULATION, "0")
        MODULATION = MODULATION + 1
        time.sleep(.5)
        SocketSendVariable(IPSEND, PORT, MODULATION, str(heart))
        MODULATION = MODULATION + 1
        time.sleep(.5)
        state = -1
    elif (state == 1):
        temperature = Temperature()
        SocketSendVariable(IPSEND, PORT, MODULATION, "1")
        MODULATION = MODULATION + 1
        time.sleep(.5)
        SocketSendVariable(IPSEND, PORT, MODULATION, str(temperature))
        MODULATION = MODULATION + 1
        time.sleep(.5)
        state = -1
    elif (state == 3):
        # todo: create a function photo_capture()

        start_t = timer()
        photo_audio_capture()
        end_t = timer()
        print("Photo taking takes:", end_t - start_t)
        state = -1
    elif (state == 4):
        SocketSendVariable(IPSEND, PORT, MODULATION, "4")
        MODULATION = MODULATION + 1
        time.sleep(.5)
        SocketSendVariable(IPSEND, PORT, MODULATION, str(step))
        MODULATION = MODULATION + 1
        time.sleep(.5)
        state = -1
    elif (state == 5):
        SocketSendVariable(IPSEND, PORT, MODULATION, "5")
        MODULATION = MODULATION + 1
        time.sleep(.5)
        # These lines are for local implementation without database
        # comment the lines before state=-1 if website is up
        SocketReceiveFile(IPRECEIVE, PORT, MODULATION)
        MODULATION = MODULATION + 1
        time.sleep(.5)
        file = "Feedback.wav"
        Speaker(file)
        state = -1

    # motion_data_saver()
    #state = STATE_CAMERA
    now = datetime.now()
    dt_string = now.strftime(DATE_TIME_FORMAT)
    print("Date and time =", dt_string)

    current_time = dt_string
    # audio_capture(current_time)
    # photo_capture(current_time)
    # break
    t_photo = threading.Thread(target=photo_capture, args=[current_time])
    # t_audio = threading.Thread(target=audio_capture, args=[current_time])

    t_photo.start()
    # t_audio.start()

    t_photo.join(6)
    # t_audio.join(7) # in seconds
    time.sleep(10)


#    (x, y, z) = accelerometer.acceleration
#
#    acc = 0
#    acc = math.sqrt(x ** 2 + y ** 2 + (z) ** 2)
#    print("Accelerometer: x:", x, " y:", y, " z:", z, " Mag:", acc)
#
#    acc = round(acc, 4)
#
#    activeTime = activeTime + 1
#    #print("Active Time:", activeTime)
#    time.sleep(1)
#
#
#    if (acc > 9.1):
#        state = STATE_CAMERA
#    elif (acc > 20):
#        step = step + 1

