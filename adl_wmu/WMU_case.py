import socket
from SocketReceive import *
from SocketSend import *
from SocketHandler import *
import time
from picamera import PiCamera
import pyaudio
from Speaker import Speaker
from gpiozero import LED
import numpy as np
import time, wave, datetime, os, csv
from gpiozero import Button
from Heartbeat import ReadHeartbeat
from getHeartbeat import getHeartbeat
from Temperature import Temperature
import board
import busio
import adafruit_adxl34x
import math

from datetime import datetime
from datetime import timedelta
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

WAV_PATH = "/home/pi/Desktop/data/audio/"
CAM_PATH = "/home/pi/Desktop/data/images/"
MO_PATH = "/home/pi/Desktop/data/motion/"

IMAGE_CNT = 10
AUDIO_DURATION = 1

DATE_TIME_FORMAT = '%Y%m%d%H%M%S'

g_picamera = PiCamera()
g_picamera.resolution = (640, 480)
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
button = Button(23)
step = 0


def util_mkdir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except Exception as e:
        print(e)
        return -1

    return 0

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

        socket_image_sending_handler(IPSEND, PORT, cnt, current_time, image_file)

        cnt = cnt + 1
        # print(image_file)

    wav_file = WAV_PATH + current_time + '/' + 'recorded.wav'
    socket_audio_sending_handler(IPSEND, PORT, current_time, wav_file)
    end_t = timer()
    print("Sending files takes:", end_t - start_t)

    return ''


def photo_capture():
    # capture 10 images for the activities
    # resolution: 1920* 1080
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

    while (cnt < IMAGE_CNT):
        image_file = image_dir + 'image' + str(cnt) + '.jpg'
        # camera.start_preview()
        camera.capture(image_file)
        # camera.stop_preview()

        cnt = cnt + 1

    # SocketSendVariable(IPSEND, PORT, MODULATION, "3")
    return ''


def audio_capture(current_time):
    # SocketSendVariable(IPSEND, PORT, MODULATION, "2")
    global MODULATION
    MODULATION = MODULATION + 1
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
        
        # record 1 seconds motion data
        if duration > 10:
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


while (True):

    # default event
    if activeTime % 10 == 0: 
        pass
        #temperature_handler(IPSEND, PORT)

    if button.is_pressed:
        SocketSendVariable(IPSEND, PORT, MODULATION, "2")
        MODULATION = MODULATION + 1
        time.sleep(.5)
        stream, audio = pyserial_start()  # start the pyaudio stream
        record_length = 5  # seconds to record
        # input('Press Enter to Record Noise (Keep Quiet!)')
        noise_chunks, _, _ = data_grabber(CHUNK / samp_rate)  # grab the data
        blue.on()
        # input('Press Enter to Record Data (Turn Freq. Generator On)')
        data_chunks, data_frames, t_0 = data_grabber(record_length)  # grab the data
        blue.off()
        data_saver(t_0)  # save the data as a .wav file
        pyserial_end()  # close the stream/pyaudio connection

        SocketSendFile(IPSEND, PORT, MODULATION, '/home/pi/Desktop/datarecorded.wav')
        MODULATION = MODULATION + 1
        time.sleep(.5)
        SocketReceiveFile(IPRECEIVE, PORT, MODULATION)
        MODULATION = MODULATION + 1
        time.sleep(.5)
        file = "Feedback.wav"
        Speaker(file)

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


    t_photo_audio = threading.Thread(target=photo_audio_capture)
    t_motion = threading.Thread(target=motion_data_saver)

    t_photo_audio.start()
    t_motion.start()

    t_photo_audio.join(6)
    t_motion.join(7) # in seconds


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


