# Author: Fei
# Date: 05/14/2021
# Brief: Resize the image using PIL
# Reference: 1) https://pillow.readthedocs.io/en/stable/reference/Image.html
#            2) Pil image resize source code: https://pillow.readthedocs.io/en/stable/_modules/PIL/Image.html#Image.resize

#Import required Image library
from PIL import Image
import os 

import timeit
import time

from timeit import default_timer as timer

IMAGE_DIR = "/home/ascc/Desktop/dataset/Images_bedroom/"
IMAGE_DIR = "/home/ascc/Desktop/dataset/Images_kitchen/"
IMAGE_DIR = "/home/ascc/Desktop/dataset/Images_LivingTV/"
IMAGE_DIR = "/home/ascc/Desktop/dataset/Images_Bathroom/"
IMAGE_DIR = "/home/ascc/LF_Workspace/Motion-Trigered-Activity/home_room_classification/keras-image-room-clasification/data_activity_0217/data_0808/images/"
IMAGE_DIR = "/home/ascc/Desktop/adl_0815/adl_0815_last7acts/images/"

IMAGE_SAVE_DIR = "/home/ascc/Desktop/Workspace/FoodRecognition_code/ImageProcess/AppleImage_ROTATE/"

IMAGE_TEST = "/home/ascc/Desktop/Workspace/FoodRecognition_code/ImageProcess/Apple/frame000003.jpg"

def image_rotate(image_file = IMAGE_TEST):
    # print("Start : %s" % time.ctime())
    start = timer()



    # Giving The Original image Directory
    # Specified
    Original_Image = Image.open(image_file)
    
    # Rotate Image By 90 Degree
    # rotated_image1 = Original_Image.rotate(90)
    
    # This is Alternative Syntax To Rotate
    # The Image
    rotated_image2 = Original_Image.transpose(Image.ROTATE_90)
    

    # rotated_image2.show()


    end = timer()
    print("Resizing the image takes:")
    print(end - start)  
    # 0.060595470014959574 t1_original
    # print("End : %s" % time.ctime())



    #Save the cropped image
    save_file = image_file
    save_file = save_file.rstrip('.jpg')
    rotated_image2.save(save_file + '_' + 'rotate' + '.jpg')


def image_rotate2(image_file, rotate_file):
    # Giving The Original image Directory
    # Specified
    Original_Image = Image.open(image_file)
    
    # Rotate Image By 90 Degree
    # rotated_image1 = Original_Image.rotate(90)
    
    # This is Alternative Syntax To Rotate
    # The Image
    rotated_image2 = Original_Image.transpose(Image.ROTATE_180)
    # rotated_image2.show()

    #Save the cropped image
    save_file = rotate_file

    rotated_image2.save(save_file)


def get_file_count_of_dir(dir, prefix=''):
    path = dir
    count = 0
    for fn in os.listdir(path):
        if os.path.isfile(dir + '/' + fn):
            if prefix != '':
                if prefix in fn:
                    count = count + 1
            else:
                print(fn)
                count = count + 1
        else:
            print('fn:', fn)
    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count

def get_rotate_file_of_dir(dir, rotate_dir):
    path = dir
    count = 0
    prefix = ''
    for fn in os.listdir(path):
        if os.path.isfile(dir + '/' + fn):
            if prefix != '':
                if prefix in fn:
                    count = count + 1
            else:
                print(fn)
                image_file = dir + '/' + fn
                rotate_file = rotate_dir + '/' + fn.strip('.jpg') + '_rotate.jpg'
                try:
                    image_rotate2(image_file, rotate_file)
                except Exception as e:
                    print(e)
                    pass
                count = count + 1
        else:
            print('fn:', fn)
    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count

def get_rotate_file_from_dir(dir):
    path = dir
    count = 0
    prefix = ''
    for fn in os.listdir(path):
        if os.path.isdir(dir + '/' + fn):
            if prefix != '':
                if prefix in fn:
                    count = count + 1
            else:
                print(fn)

                source_dir = dir + '/' + fn
                rotate_dir = source_dir + '_rotate'
                print("rotate_dir:", rotate_dir)
                if source_dir.find('rotate') > -1:
                    continue
                
                try:
                    if os.path.exists(rotate_dir) == False:
                        os.mkdir(rotate_dir)
                except Exception as e:
                    print(e)
                    pass

                count = get_rotate_file_of_dir(source_dir, rotate_dir) + count

        else:
            print('fn:', fn)


    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count

#image_rotate(IMAGE_TEST)

# count = get_file_count_of_dir(IMAGE_DIR)
# count = get_rotate_file_of_dir(IMAGE_DIR, IMAGE_SAVE_DIR)
IMAGE_DIR = "/home/ascc/Desktop/image_living_bedroom/new_data/images/"
count = get_rotate_file_from_dir(IMAGE_DIR)
print("Rotate file count:", count)
