# program to capture single image from webcam in python
  
# importing OpenCV library
# from cv2 import *
import cv2 as cv
import re
import time  
from timeit import default_timer as timer
# initialize the camera
# If you have multiple camera connected with 
# current device, assign a value in cam_port 
# variable according to that

"""
ascc@robot:~/lf_workspace/IROS23/robot_code$ ll /dev/video*
crw-rw----+ 1 root plugdev 81, 0 Feb  5 19:07 /dev/video0
crw-rw----+ 1 root plugdev 81, 1 Feb  5 19:07 /dev/video1
crw-rw----+ 1 root video   81, 2 Feb  5 19:08 /dev/video2

"""
source = 0
w = 224
h = 224

PHOTO_IMAGE_POST_FILE_NAME = '/home/ascc/asccbot_v3/asccChatBot/AsccChatbot/Images/photo_quickly_taking_post.config'
CAM_PATH = '/home/ascc/lf_workspace/IROS23/data/image/'
CAMERA_INIT_TIME = 2 # Seconds for initialing the camera or you may get a green photo

def getCurrentTimestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def store_name_to_file(file_name, name):
    with open(file_name, 'w') as f:
        f.write(name)
        f.close()
    return ''


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

def test_cam():
    caps = list(map(create_capture, [0]))

    shot_idx = 0
    image_cnt = 3
    while True:
        imgs = []
        for i, cap in enumerate(caps):
            ret, img = cap.read()
            imgs.append(img)
            cv.imshow('capture %d' % i, img)

        ch = cv.waitKey(1)
        if ch == 27: 
            break
        if ch == ord(' '):
            for i, img in enumerate(imgs):
                fn = '%s/shot_%d_%03d.jpg' % (CAM_PATH, i, shot_idx)
                cv.imwrite(fn, img)
                store_name_to_file(PHOTO_IMAGE_POST_FILE_NAME, fn)
                print(fn, 'saved')
            shot_idx += 1
            if shot_idx == image_cnt:
                cv.destroyAllWindows()
                break


def test_cam2():
    caps = list(map(create_capture, [0]))

    shot_idx = 0
    image_cnt = 3
    start_t = timer()
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
                fn = '%s/shot_%d_%03d.jpg' % (CAM_PATH, i, shot_idx)
                cv.imwrite(fn, img)
                store_name_to_file(PHOTO_IMAGE_POST_FILE_NAME, fn)
                print(fn, 'saved')
            shot_idx += 1
            if shot_idx == image_cnt:
                # cv.destroyAllWindows()
                break
            


while True:
    test_cam2()
    time.sleep(5)



