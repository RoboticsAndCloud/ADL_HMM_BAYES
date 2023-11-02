STATE_HEARTBEAT = 0
STATE_TEMPERATURE = 1
STATE_AUDIO = 2
STATE_CAMERA = 3
STATE_STEP = 4
STATE_TEXT_2_SPEECH = 5
STATE_ACTIVITY_TRIGGER = 6
STATE_ACTIVITY_TRIGGER_IMAGE = 7
STATE_ACTIVITY_TRIGGER_AUDIO = 8


STATE_ENV_ACTIVITY_CMD_TAKING_IMAGE = 10
STATE_ENV_ACTIVITY_CMD_TAKING_AUDIO = 11
STATE_ENV_ACTIVITY_CMD_TAKING_MOTION = 12
STATE_ENV_ACTIVITY_CMD_TAKING_FUSION = 13

# STATE_MEDICATION_ACTIVITY_CMD_TAKING_IMAGE = 10
# STATE_MEDICATION_ACTIVITY_CMD_PLAY_AUDIO = 11

STATE_ADL_ACTIVITY_WMU_AUDIO = 15
STATE_ADL_ACTIVITY_WMU_IMAGE = 16
STATE_ADL_ACTIVITY_WMU_MOTION = 17

STATE_ADL_ACTIVITY_ROBOT_AUDIO = 25
STATE_ADL_ACTIVITY_ROBOT_IMAGE = 26

WMU_MOTION_FILE_NOTIFICATION_FILE = '/home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES/room_motion_activity/ascc_data/notice.txt'
WMU_AUDIO_FILE_NOTIFICATION_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/ascc_data/notice.txt'
WMU_IMAGE_FILE_NOTIFICATION_FILE = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/ascc_data/notice.txt'


# MOTION_FILE_SAVED_FOLDER = '/home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/ascc_data/motion/'
# AUDIO_FILE_SAVED_FOLDER = '/home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/ascc_data/audio/'
# IMAGE_FILE_SAVED_FOLDER = '/home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/ascc_data/image/'
# IMAGE_FILE_SAVED_FOLDER = '/home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_Monitoring_Web/website/public/image/'

MOTION_FILE_SAVED_FOLDER = '/home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/ascc_data/09icra_real_test/motion/'
AUDIO_FILE_SAVED_FOLDER = '/home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/ascc_data/09icra_real_test/audio/'
IMAGE_FILE_SAVED_FOLDER = '/home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/ascc_data/09icra_real_test/image/'
# IMAGE_FILE_SAVED_FOLDER = '/home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_Monitoring_Web/website/public/image/'



DATE_TIME_FORMAT = '%Y%m%d%H%M%S'


# WMU
WMU_IPRECEIVE = "192.168.1.131" 
WMU_IPRECEIVE = "192.168.1.127" # watch
WMU_IPRECEIVE = "192.168.0.111" # watch, TP-LINK


# WMU_IPSEND = "10.227.99.196"
WMU_IPSEND = '192.168.1.134'
#WMU_IPRECEIVE = "192.168.1.131"
#WMU_IPSEND = "192.168.1.126"
WMU_RECEIVE_PORT = 59100 # WMU server
#WMU_RECEIVE_PORT = 8080 # WMU server

WMU_SEND_PORT = 59000 # robot server

WMU_COMPANION_ROBOT_IPRECEIVE = "192.168.0.107" #TP-LINK, the companion robot 
WMU_COMPANION_ROBOT_RECEIVE_PORT = 59101 # Companion robot port

ROBOT_IP = WMU_IPSEND 
ROBOT_PORT = WMU_SEND_PORT # robot server

