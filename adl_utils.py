#!/usr/bin/python
################################################################################ #
# Copyright (c) 2020 ASCC LAB, OSU. All Rights Reserved
# ################################################################################
"""
This module provide configure file management service in i18n environment.
Authors: fei(fei.liang@okstate.edu)
Date: 2020/05/25 17:23:06
"""
import locale
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText

from email import encoders

import log
import logging
import random
import signal
import subprocess
import sys
import os
import time

# from google.cloud import texttospeech

import requests


#from google.cloud import speech_v1
#from google.cloud.speech_v1 import enums
# from google.cloud import speech
#from google.cloud.speech import enums
import io

# import the module
import asyncio

TTS_AUDIO_FILE_DIR = "/home/ascc/asccbot_v3/medication_system/data/robot_generate_audio/"

TTS_AUDIO_FILE_NAME = "/home/ascc/asccbot_v3/wearable_device_new_design/server/audio_response/ttsRes.wav"

TTS_AUDIO_FORMAT_FILE_NAME = "/home/ascc/asccbot_v3/wearable_device_new_design/server/audio_response/ttsResFormat.wav"

IMAGE_DIR = "/home/ascc/asccbot_v3/wearable_device_new_design/server/images/"



def get_latest_image_file_name(dir=IMAGE_DIR, file_prefix='cap'):
    _prefix = file_prefix

    # ls /home/ascc/asccbot_v3/wearable_device_new_design/server/audio/ | grep 'cap' | tail -n 1
    cmd = 'ls -tr ' + IMAGE_DIR + ' ' + '|' + ' ' + 'grep' + ' ' + _prefix + '| tail -n 1'
    audio_file_name = ''
    try:
        # audio_file_name = os.system(cmd)
        audio_file_name = os.popen(cmd).read()
    except Exception as e:
        raise e
    #print(audio_file_name)
    audio_file_name = ''.join([dir, audio_file_name])
    res = audio_file_name.rstrip()
    return res



class ActionAudioNode():

    def __init__(self):
        log.init_log(log.default_log_dir)  # ./log/my_program.log./log/my_program.log.wf7

        # # cap_Fri_Aug_27_12:43:52_2021.wav
        self.audio_file_count_init = utils.get_file_count_of_dir(dir=AUDIO_FILE_DIR, prefix='cap')
        logging.info("Initial Audio file count %d", self.audio_file_count_init)

    def run(self):

        while True:
            self.check_and_process_audio()


        # ## TODO get chat_id based on the received message, son's id, daughter's id
        #

## ffmpeg -i resAudio.wav -acodec pcm_s16le -ac 1 -ar 16000 out.wav
## ffmpeg -i resAudio.wav -acodec pcm_s16le -ac 1 -ar 16000 out.wav



# ffmpeg -i /home/ascc/asccbot_v3/wearable_device_new_design/server/audio/cap_Sun_Aug_29_11:49:19_2021.wav -acodec pcm_s16le -ac 1 -ar 16000 /home/ascc/asccbot_v3/wearable_device_new_design/server/audio/cap_Sun_Aug_29_11:49:19_2021_Format.wav



def format_audio_file(input_file):

    dest_format_file = input_file.split('/')[-1]
    dest_format_file = dest_format_file.split('.wav')[0] + 'Format.wav'
    #format_cmd = 'ffmpeg -i ' + input_file + ' ' + '-acodec pcm_s16le -ac 1 -ar 16000' + ' ' + TTS_AUDIO_FORMAT_FILE_NAME + ' -y'

    dest_format_file = ''.join([TTS_AUDIO_FILE_DIR, dest_format_file])

    format_cmd = 'ffmpeg -i ' + input_file + ' ' + '-acodec pcm_s16le -ac 1 -ar 16000' + ' ' + dest_format_file + ' -y'

    res = dest_format_file


    cp_cmd = 'cp ' + dest_format_file + ' ' + TTS_AUDIO_FORMAT_FILE_NAME

    print(format_cmd)
    print(cp_cmd)
    try:
        os.system(format_cmd)

        if cp_cmd.find('ttsRescap_') > -1:
            os.system(cp_cmd)
            print("################## COPY CMD ########################### cmd: %s", cp_cmd)

    except:
        res = ''

    return res

def genearte_tts_audio_file_name(dir=TTS_AUDIO_FILE_DIR, prefix='ttsRes', dest_file = ''):
    dest_prefix = ''
    if dest_file != '':
        dest_prefix = dest_file.split('/')[-1]
        dest_prefix = dest_prefix.split('.wav')[0]

    audio_file_name = ''.join([dir, prefix, dest_prefix, '.wav'])
    return audio_file_name

def genearte_tts_audio_file_name_by_file_count(dir=TTS_AUDIO_FILE_DIR, prefix='ttsRes'):
    audio_prefix = prefix
    _num = 0
    try:
        _num = get_file_count_of_dir(dir, audio_prefix)
    except Exception as e:
        raise e

    audio_file_name = ''.join([dir, audio_prefix, str(_num), '.wav'])
    return audio_file_name


def store_name_to_file(file_name, name):
    with open(file_name, 'w') as f:
        f.write(name)
        f.close()
    return ''

"""
Brief: get file count of a director

Raises:
     NotImplementedError
     FileNotFoundError
"""
def get_file_count_of_dir(dir, prefix=''):
    path = dir
    count = 0
    for fn in os.listdir(path):
        if os.path.isfile(dir + '/' + fn):
            if prefix != '':
                if prefix in fn:
                    count = count + 1
            else:
                count = count + 1
        else:
            print('fn:', fn)
    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count

"""
Brief: get file list of a director

Raises:
     NotImplementedError
     FileNotFoundError
"""
def get_file_list_of_dir(dir, prefix=''):
    path = dir
    count = 0
    file_list = []
    for fn in os.listdir(path):
        file_path = dir + '/' + fn
        if os.path.isfile(file_path):
            if prefix != '':
                if prefix in fn:
                    file_list.append(file_path)
            else:
                file_list.append(file_path)
        else:
            print('fn:', fn)

    return file_list


def download_image(dir, url, image):
    if not os.path.exists(dir):
        os.makedirs(dir)

    # move to new directory
    os.chdir(dir)

    # writing images
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(image, 'wb') as f:
                f.write(requests.get(url).content)
                f.close()
    except:
        return False

    return True

def read_name_from_file(file_name):
    name = ''
    with open(file_name, 'r') as f:
        name = str(f.read().strip())
        f.close()
    #logging.info('name:%s', name)
    return name


def read_dir_name(file_name):
    with open(file_name, 'r') as f:
        dir_name = str(f.read().strip())
        f.close()
    print('dir_name:%s', dir_name)
    return dir_name


def write_res_into_file(file_name, res):
    with open(file_name, 'w') as f:
        f.write(str(res))
        f.close()

    return True

#
# class ImageDisplayOnRobotFace(object):
#     def __init__(self):
#         self.display_cmd = constants.DISPLAY_IMAGE_CMD
#         self.display_process = 'xxx'
#         return
#
#     def start_display_image(self):
#
#         # if tools.utils_linux.isRunning(self.record_process):
#         #     print('vidoe record process is running')
#         #     logging.warn("vidoe record process is running")
#         #     return
#
#         self.display = subprocess.Popen([self.display_cmd], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
# #        self.video = os.system(self.record_cmd)
#         print((("start_display_image started")))
#
# class PickNumberImageDisplayOnRobotFace(object):
#     def __init__(self):
#         self.display_cmd = constants.PICK_NUMBER_CMD
#         self.display_process = 'pickxxx'
#         return
#
#     def start_pick_and_display_image(self):
#
#         # if tools.utils_linux.isRunning(self.record_process):
#         #     print('vidoe record process is running')
#         #     logging.warn("vidoe record process is running")
#         #     return
#
#         self.display = subprocess.Popen([self.display_cmd], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
# #        self.video = os.system(self.record_cmd)
#         print((("start_pick_number_display_image started")))
#
# class TwitterMonitorTools():
#     def __init__(self):
#         log.init_log(log.default_log_dir)  # ./log/my_program.log./log/my_program.log.wf7
#         consumer_key = '3SIoQh3uY92ebnRRTz4GeDA6B'
#         consumer_secret = 'Uzv7Hs3WHmZd3PKlwkqg7DyJUcnMrKOJaWVBjvmt8vHBbXl5uv'
#         access_token = '1195185859374600192-62EA2PO59AT3TdMYlL1tH6X8FRqo5F'
#         access_token_secret = 'wDcGvVY22mjafOuGwliPn95bYi2UUE5pAxzJFj5taOHdS'
#         auth = tp.OAuthHandler(consumer_key, consumer_secret)
#         auth.set_access_token(access_token, access_token_secret)
#         self.api = tp.API(auth)
#         self.latest_id = 1
#
#     def twitter_api():
#         # credentials to login to twitter api
#         consumer_key = '3SIoQh3uY92ebnRRTz4GeDA6B'
#         consumer_secret = 'Uzv7Hs3WHmZd3PKlwkqg7DyJUcnMrKOJaWVBjvmt8vHBbXl5uv'
#         access_token = '1195185859374600192-62EA2PO59AT3TdMYlL1tH6X8FRqo5F'
#         access_token_secret = 'wDcGvVY22mjafOuGwliPn95bYi2UUE5pAxzJFj5taOHdS'
#         auth = tp.OAuthHandler(consumer_key, consumer_secret)
#         auth.set_access_token(access_token, access_token_secret)
#         api = tp.API(auth)
#         return api
#
#     def store_twitter_photo_name(self, file_name, photo_name):
#         with open(file_name, 'w') as f:
#             f.write(photo_name)
#             f.close()
#         logging.info('twitter_photo_name:%s', photo_name)
#         return ''
#
#     def read_twitter_photo_name(self, file_name):
#         with open(file_name, 'r') as f:
#             photo_name = str(f.read().strip())
#             f.close()
#         logging.info('twitter_photo_name:%s', photo_name)
#         return photo_name
#
#     def read_twitter_last_mentioned_id(self, file_name):
#         with open(file_name, 'r') as f:
#             last_id = int(f.read().strip())
#             f.close()
#         logging.info('read_twitter_last_mentioned_id:%s', last_id)
#         return last_id
#
#     def moni_task(self):
#         # run
#         print('moni_task Running')
#         while True:
#             self.latest_id = self.read_twitter_last_mentioned_id(twitter_mentioned_id_filename)
#             self.get_mentions_tweet(self.latest_id)
#             time.sleep(10)
#             print('moni_task Running')
#
#     def callback_received_stt(self,msgs):
#         self.stt_result = msgs.data
#         self.has_stt = True
#
#     def display(self):
#         twitter_mention = 'You are mentioned on twitter, please watch the screen'
#
#         #./ stt.py: self.stt_text = rospy.Publisher('stt_text', String, queue_size=10)
#
#         #self.received_stt = rospy.Subscriber('tele_text', String, self.callback_received_stt)
#         # time.sleep(1.5)
#         # self.stt_result = ''
#         # self.start_stt.publish("start")
#         # a_time = time.time()
#         # while (self.stt_result == ''):  ###need to improve
#         #     # if no reponse in 5 seconds, break
#         #     if ((time.time() - a_time) > 10):
#         #         tts("you speak nothing, I will ask you later")
#         #         for a_user in total_user_id:
#         #             a_need_md = []
#         #             for i in needed_item:
#         #                 if i['user_id'] == a_user:
#         #                     a_need_md.append(i['medication_name'])
#         #             new_medicine_dict = []
#         #             for sub_json in medicine_dict:
#         #                 if sub_json['user_id'] == a_user:
#         #                     if (sub_json['medication_name'] in a_need_md) and (sub_json['remind_time'] == get_rmd_time):
#         #                         sub_json['remind_time'] = \
#         #                         time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + seconds_later)).split(
#         #                             ' ')[1]
#         #                     new_medicine_dict.append(sub_json)
#         #             with open(json_dir + a_user + '_for_use.json', 'w') as f:
#         #                 f.write(json.dumps(new_medicine_dict))
#         #         self.has_task = False
#         #         return 0
#
#         TTSTool.tts(twitter_mention)
#         imageDisplay = ImageDisplayOnRobotFace()
#         imageDisplay.start_display_image()
#         return
#
#     def get_mentions_tweet(self, since_id):
#         mentions_tweets = self.api.mentions_timeline(since_id=since_id)
#         logging.info('get_mentions_tweet, since_id:%s', since_id)
#
#         last_tweet_id = []
#         flag = 0
#         for tweet in mentions_tweets:
#             # print(tweet)
#             id = str(tweet.id)
#             last_tweet_id.append(tweet.id)
#             text = tweet.text
#             print(text)
#             user_name = tweet.user.name
#             print(user_name)
#             user_description = tweet.user.description
#             mention_time = str(tweet.created_at).replace(" ", "")
#             print(mention_time)
#
#             logging.info('username: %s, tweet.id:%s, text:%s', user_name, id, text)
#             msg_type = 'text'
#
#             # todo
#             if 'media' in tweet.entities.keys():
#                 # print(tweet.entities['media'])
#                 for media_item in tweet.entities['media']:
#                     if media_item['type'] == 'photo':
#                         msg_type = 'photo'
#                         print(media_item['media_url'])
#                         image_name = user_name + str(mention_time) + '.jpg'
#                         download_image(dir=twtter_image_download_dir, url=media_item['media_url'],
#                                        image=image_name)
#                         image_name = twtter_image_download_dir + '/' + image_name
#                         self.store_twitter_photo_name(twitter_image_display_file_name, image_name)
#
#                         ## todo display image on screen
#                         flag = flag + 1
#                         try:
#                             #if flag > 1:
#                             #    break
#                             self.display()
#
#                             time.sleep(30)
#                         except Exception as e:
#                             print('display error:' + str(e))
#                             logging.error(str(e))
#
#
#             if msg_type == 'text':
#                 twitter_mention = 'You are mentioned on twitter'
#                 #ascc
#                 # TTSTool.tts(twitter_mention)
#                 print(text)
#
#         # get latest id
#         print(last_tweet_id)
#         # self.latest_id = last_tweet_id[0]
#         if len(last_tweet_id) == 0:
#             return
#         self.store_twitter_photo_name(twitter_mentioned_id_filename, str(last_tweet_id[0]))
#         print('lastidread:')
#         print(self.read_twitter_last_mentioned_id(twitter_mentioned_id_filename))
#
#     def tweet_image(message, image):
#         api = TwitterMonitorTools.twitter_api()
#         filename = image
#         api.update_with_media(filename, status=message)
#
#
# class VideoRecording(object):
#     def __init__(self):
#         self.record_cmd = constants.VIDEO_CMD
#         self.record_process = 'record_video.py'
#         return
#
#     def start_video_recording(self):
#
#         if tools.utils_linux.isRunning(self.record_process):
#             print('vidoe record process is running')
#             logging.warn("vidoe record process is running")
#             return
#
#         photoTaking = PhotoTaking()
#         try:
#             photoTaking.stop_camera()
#         except Exception as e:
#             logging.warn("stop_camera failed, warning:" + str(e))
#
#         self.video = subprocess.Popen([self.record_cmd], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
# #        self.video = os.system(self.record_cmd)
#         print((("video_recording started")))
#
#     def stop_video_recording(self):
#         #os.killpg(os.getpgid(self.video.pid), signal.SIGTERM)
#         #self.video.terminate()
#         #self.video.kill()
#         print("stop_video_recording")
#
# class PhotoTaking(object):
#     def __init__(self):
#         self.src_folder = constants.ASCCV2_SRC_FOLDER
#         self.camera_process = 'realsense_camera/F200Nodele'
#         return
#
#     def start_photo(self):
#
#         if not tools.utils_linux.isRunning(self.camera_process):
#              self.start_camera()
#         self.photo = subprocess.Popen([self.src_folder + "shell_scripts/photo.sh"], stdout=subprocess.PIPE,
#                                       shell=True, preexec_fn=os.setsid)
#         print((("Photo-taking started")))
#         time.sleep(1)
#
#     def start_photo_quickly(self):
#
#         # if not tools.utils_linux.isRunning(self.camera_process):
#         #      self.start_camera()
#         self.photo = subprocess.Popen([self.src_folder + "shell_scripts/photo_quick_taking.sh"], stdout=subprocess.PIPE,
#                                       shell=True, preexec_fn=os.setsid)
#         print((("Photo-taking_quick started")))
#         time.sleep(1)
#
#     def start_photo_taking_by_wearable_camera(self):
#
#         # if not tools.utils_linux.isRunning(self.camera_process):
#         #      self.start_camera()
#         self.photo = subprocess.Popen([self.src_folder + "shell_scripts/photo_quick_taking_by_wearable_device.sh"], stdout=subprocess.PIPE,
#                                       shell=True, preexec_fn=os.setsid)
#         print((("Photo-taking_by_wearable_camera started")))
#         time.sleep(1)
#
#     def stop_photo(self):
#         os.system("rosnode kill /VideoServer")
#         os.system("rosnode kill /Photo")
#         os.system("rosnode kill /Photo_Controller")
#         # if not self.recording:
#         #     self.stop_camera()
#         time.sleep(1)
#         os.killpg(os.getpgid(self.photo.pid), signal.SIGTERM)
#         self.photo.terminate()
#         self.photo.kill()
#         print((("Photo-taking Stopped")))
#
#     def start_camera(self):
#         try:
#             self.stop_camera()
#         except:
#             pass
#         self.camera = subprocess.Popen([self.src_folder + "shell_scripts/camera.sh"], stdout=subprocess.PIPE,
#                                        shell=True, preexec_fn=os.setsid)
#
#     def stop_camera(self):
#         os.system("rosnode kill /nodelet_manager")
#         self.camera.terminate()
#         self.camera.kill()
#         time.sleep(1)
#         os.killpg(os.getpgid(self.camera.pid), signal.SIGTERM)
#

class TTSTool(object):
    """Text to Speech tool. User can use TTSTool.tts() to Convert text to speech
        Reference: Google SPEECH API
    """

    def __init__(self):
        # rospy.on_shutdown(self.cleanup)
        return

    def tts(input_text, dest_file = ''):
        """
        Text to Speech, create reminder.mp3 and save into current dir
        :return: reminder.mp3
        """
        res = ''
        print('in tts: ', dest_file, input_text)
        res = genearte_tts_audio_file_name(dest_file = dest_file)
        print("genearte_tts_audio_file_name:", res)

        try:
            client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=input_text)
            voice = texttospeech.VoiceSelectionParams(
                language_code='en-US',
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16)

            # response = client.synthesize_speech(synthesis_input, voice, audio_config)

            response = client.synthesize_speech(
                input = synthesis_input,
                voice = voice,
                audio_config = audio_config
            )

            with open(res, 'wb') as out:
                out.write(response.audio_content)

            # os.system('mplayer ' + res)
        except Exception as e:
            res = str(e)
            print(e)


        return res

    def stt(local_audio_file):
        """
        Transcribe a short audio file using synchronous speech recognition

        Args:
          local_file_path Path to local audio file, e.g. /path/audio.wav
        """

        client = speech.SpeechClient()

        # local_file_path = 'resources/brooklyn_bridge.raw'

        # The language of the supplied audio
        language_code = "en-US"

        # Sample rate in Hertz of the audio data sent
        sample_rate_hertz = 16000
        # sample_rate_hertz = 41000


        # Encoding of audio data sent. This sample sets this explicitly.
        # This field is optional for FLAC and WAV audio formats.
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        config = {
            "language_code": language_code,
            "sample_rate_hertz": sample_rate_hertz,
            # "encoding": encoding,
        }
        with io.open(local_audio_file, "rb") as f:
            content = f.read()
        audio = {"content": content}

        response = client.recognize(config=config, audio=audio)

        res = ''
        for result in response.results:
            # First alternative is the most probable result
            alternative = result.alternatives[0]
            print(u"Transcript: {}".format(alternative.transcript))
            res = alternative.transcript
            break

        return res


    def post_data_to_rasa(rasa_api, msg, sender='default'):
        #INSERT WEBHOOK API URL HERE
        rest_rasa_api = rasa_api
        messages = msg

        # Speak message to user and save the response
        # If user doesn't respond, quietly stop, allowing user to resume later
        if messages is None:
            return
        # Else reset messages

        # Send post requests to said endpoint using the below format.
        # "sender" is used to keep track of dialog streams for different users
        try:
            data = requests.post(
                rest_rasa_api, json={"message": messages, "sender": sender}
            )
        except Exception as e:
            print(e)
            return None
        # A JSON Array Object is returned: each element has a user field along
        # with a text, image, or other resource field signifying the output
        # print(json.dumps(data.json(), indent=2))
        print(data)
        messages = []
        try:
            for next_response in data.json():
            # print(next_response)
                if "text" in next_response:
                    messages.append(next_response["text"])
            # Output all but one of the Rasa dialogs
        except Exception as e:
            print("message got error " + str(e))
            return None
        if len(messages) >= 1:
            for rasa_message in messages:
                print(rasa_message)

        # Kills code when Rasa stop responding
        if len(messages) == 0:
            messages = ["no response from rasa"]
            return

        # Allows a stream of user inputs by re-calling query_rasa recursively
        # It will only stop when either user or Rasa stops providing data
        return messages


class HttpTool(object):
    """Text to Speech tool. User can use TTSTool.tts() to Convert text to speech
        Reference: Google SPEECH API
    """

    def __init__(self):
        # rospy.on_shutdown(self.cleanup)
        return

    def tts(input_text):
        """
        Text to Speech, create reminder.mp3 and save into current dir
        :return: reminder.mp3
        """
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.types.SynthesisInput(text=input_text)
        voice = texttospeech.types.VoiceSelectionParams(
            language_code='en-US',
            ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)
        audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.MP3)
        response = client.synthesize_speech(synthesis_input, voice, audio_config)
        with open('reminder.mp3', 'wb') as out:
            out.write(response.audio_content)
        os.system('mplayer reminder.mp3')

    def post_data_to_rasa(rasa_api, json_msg, sender='default'):
        #INSERT WEBHOOK API URL HERE
        rest_rasa_api = rasa_api
        messages = json_msg

        # Speak message to user and save the response
        # If user doesn't respond, quietly stop, allowing user to resume later
        if messages is None:
            return
        # Else reset messages

        # Send post requests to said endpoint using the below format.
        # "sender" is used to keep track of dialog streams for different users
        try:
            data = requests.post(
                rest_rasa_api, json=messages
            )
        except Exception as e:
            print(e)
            return None
        # A JSON Array Object is returned: each element has a user field along
        # with a text, image, or other resource field signifying the output
        # print(json.dumps(data.json(), indent=2))
        print(data)
        messages = []
        try:
            for next_response in data.json():
            # print(next_response)
                if "text" in next_response:
                    messages.append(next_response["text"])
            # Output all but one of the Rasa dialogs
        except Exception as e:
            print("message got error " + str(e))
            return None
        if len(messages) >= 1:
            for rasa_message in messages:
                print(rasa_message)

        # Kills code when Rasa stop responding
        if len(messages) == 0:
            messages = ["no response from rasa"]
            return ''

        # Allows a stream of user inputs by re-calling query_rasa recursively
        # It will only stop when either user or Rasa stops providing data
        return messages

    def post_message_by_chatid_token(chat_id, token, message):

        # send a message
        chat_id = chat_id
        token = token
        request_api = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}'.format(token, chat_id)
        json_data = {
            "chat_id": chat_id,
            "text": message
        }
        return HttpTool.post_data_to_rasa(request_api, json_data)

#
# class MessageFilterTool(object):
#
#     def message_filter(msg):
#         tmpMsg = str(msg)
#
#         prefix_filter_list = ['send this message', 'send the message', 'send a message', 'send message']
#
#         TO_MY_MUM = 'to my mum'
#         TO_MY_CHILD = 'to my child'
#
#         suffix_filter_list = ['to my mum', 'to my mother', 'to my dad', 'to my grandma', 'to my grandmother',
#                               'to my grandpa',
#                               'to my child', 'to my son', 'to my daughter']
#
#         for f in prefix_filter_list:
#             if f in tmpMsg:
#                 tmpMsg = tmpMsg.replace(f, '', 1)
#                 break
#
#         for f in suffix_filter_list:
#             if f in tmpMsg:
#                 tmpMsg = tmpMsg.replace(f, '', 1)
#                 break
#
#         # if SEND_THIS_MESSAGE in tmpMsg:
#         #     tmpMsg = tmpMsg.replace(SEND_THIS_MESSAGE, '', 1)
#         # elif SEND_THE_MESSAGE in tmpMsg:
#         #     tmpMsg = tmpMsg.replace(SEND_THE_MESSAGE, '', 1)
#         #
#         # if TO_MY_MUM in tmpMsg:
#         #     tmpMsg = tmpMsg.replace(TO_MY_MUM, '', 1)
#         # elif TO_MY_CHILD in tmpMsg:
#         #     tmpMsg = tmpMsg.replace(TO_MY_CHILD, '', 1)
#
#         tmpMsg = tmpMsg.strip()
#         return tmpMsg
#
# class MusicPlayer(object):
#
#     def play_random_music():
#         musicList = [
#             'RiverFlowsinYou.mp3',
#             'Canon.mp3'
#         ]
#
#         music_name = random.choice(musicList)
#         music = music_path + music_name
#
#         try:
#             musicList = get_file_list_of_dir(music_path, prefix='mp3')
#             music_name = random.choice(musicList)
#             music = music_name
#         except:
#             pass
#
#         print(music)
#         #cmd = 'afplay' + ' ' + music
#         cmd = 'mplayer' + ' ' + music
#         print(cmd)
#         subprocess.Popen([cmd], stdout=subprocess.PIPE,
#                                        shell=True, preexec_fn=os.setsid)
#         return music
#
#     def play_music(music_name):
#         musicList = [
#             'RiverFlowsinYou.mp3',
#             'Canon.mp3'
#         ]
#
#         try:
#             musicList = get_file_list_of_dir(music_path, prefix='mp3')
#             music = music_path + music_name + '.mp3'
#         except:
#             music = music_name + '.mp3'
#             pass
#
#
#         if music in musicList:
#             music = music_path + music
#         else:
#             music = random.choice(musicList) # todo search on the internet
#
#         print(music)
#         #cmd = 'afplay' + ' ' + music
#         cmd = 'mplayer' + ' ' + music
#         subprocess.Popen([cmd], stdout=subprocess.PIPE,
#                                        shell=True, preexec_fn=os.setsid)
#         return music
#
#     def stop_music():
#
#         ## todo check whether the robot is playing music
#
#         # cmd = "ps aux |grep '/home/ascc/asccbot_v3/asccChatBot/music/data' | grep -v '/bin/sh' | grep -v 'grep' | awk '{print $2}' | xargs kill -9"
#         cmd = "ps aux |grep " + constants.MUSIC_FILE_PATH + " | grep -v '/bin/sh' | grep -v 'grep' | awk '{print $2}' | xargs kill -9"
#
#         print('stop playing music, ' + cmd)
#         subprocess.Popen([cmd], stdout=subprocess.PIPE,
#                                        shell=True, preexec_fn=os.setsid)
#         logging.info('stop playing music, ' + cmd)
#
#     def play_ding_music(music_name):
#         music = '/home/ascc/asccbot_v3/asccChatBot/AsccChatbot/Images/pickNumber/Ding1.mp3'
#         print(music)
#         #cmd = 'afplay' + ' ' + music
#         cmd = 'mplayer' + ' ' + music
#         subprocess.Popen([cmd], stdout=subprocess.PIPE,
#                                        shell=True, preexec_fn=os.setsid)
#         return music


def get_weather():
    return ''


def send_alarm():

    filename = "photo.jpg"
    filename= get_latest_image_file_name();

    fromaddr = "ascclab2020@gmail.com"
    pw = "bhwumrwsbwwfiieq"
    # self.toaddr = "zhidong.su@okstate.edu"
    toaddr = "fei.liang@okstate.edu"
    link = filename

    email = MIMEMultipart()
    email['From'] = fromaddr
    email['To'] = toaddr
    email['Subject'] = "Alarm!!! A fall detect! Photo taken by ASCCBot"
    body = "Attached is the photo taken by ASCC Wearable Device. Please check it!"
    email.attach(MIMEText(body, 'plain'))
    emailsent = False

    if send_email(fromaddr, pw, toaddr, email, filename, link):
        print("Send alarms to healthcare\n")
        logging.info("Send alarms to healthcare")
    else:
        logging.warn("Send alarms failed")


def send_email(fromaddr, pw, toaddr, msg, filename, link):
    attachment = open(link, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(part)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, pw)
    text = msg.as_string()
    try:
        server.sendmail(fromaddr, toaddr, text)
        server.quit()
        return True
    except:
        server.quit()
        return False

if __name__ == "__main__":
    import log
    # log.init_log("./log/my_program")  # ./log/my_program.log./log/my_program.log.wf7
    # logging.info("Hello World!!!")
    # print("My name is Elsa.")
    # # TTSTool.tts("My name is Elsa.")
    # # rospy.init_node('ASCCBot_MedicationReminderTimerChecker')
    #
    # rasa_api = 'http://localhost:5005/webhooks/rest/webhook'
    # rasa_api = 'https://f7388d57f110.ngrok.io/webhooks/rest/webhook'
    #
    #
    # msg = 'Hi'
    # sender = 'Elsa'
    # res = TTSTool.post_data_to_rasa(rasa_api, msg, sender)
    # if res == None:
    #     print(None)

    # testMessage = 'send a message what do you want to eat to my mum'
    # testMessage2 = "send this message 'I love yoo' to my child"
    # print(MessageFilterTool.message_filter(testMessage))
    # print(MessageFilterTool.message_filter(testMessage2))
    #
    # #vr = VideoRecording()
    # #vr.start_video_recording()
    #
    # print(get_file_list_of_dir(music_path, prefix='mp3'))
    #
    # MusicPlayer.play_random_music()
    #
    # tw = TwitterMonitorTools()
    # tw.get_mentions_tweet(1)
    # tw.moni_task()
    get_weather()

    print("My name is Elsa. over")

