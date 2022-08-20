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

IMAGE_DIR = "/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Reading/"

IMAGE_SAVE_DIR = "/home/ascc/Desktop/Workspace/FoodRecognition_code/ImageProcess/AppleImage_ROTATE/"

IMAGE_TEST = "/home/ascc/Desktop/Workspace/FoodRecognition_code/ImageProcess/Apple/frame000003.jpg"

g_copy_file_count = 0

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
    rotated_image2 = Original_Image.transpose(Image.ROTATE_90)
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


def copy_file_from_dir(dir, dest_dir):
    global g_copy_file_count

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
                dest_file = dest_dir + '/' + fn.strip('.jpg') + '-' + str(g_copy_file_count) + '.jpg'
                #dest_file = dest_dir + '/' + room_name + '-' + str(g_copy_file_count) + '.jpg'

                try:
                    cmd = 'cp ' + image_file + ' ' + dest_file
                    print(cmd)
                    os.system(cmd)
                    g_copy_file_count = g_copy_file_count + 1
                except Exception as e:
                    print(e)
                    pass
                count = count + 1
        else:
            print('fn:', fn)
    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count

def generate_copy_image_file_from_dir(dir, dest_dir):
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
                
                try:
                    if os.path.exists(dest_dir) == False:
                        os.mkdir(dest_dir)
                except Exception as e:
                    print(e)
                    pass

                count = copy_file_from_dir(source_dir, dest_dir) + count

        else:
            print('fn:', fn)


    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count


def copy_ascc_act_dir(dir, target_dir, type, copy_flag=False):
    path = dir
    count = 0
    for fn in os.listdir(path):
        if os.path.isdir(dir + '/' + fn):
            print('dir:', fn)
            # if ('-' not in ascc_date_str):
            #     continue

            # Motion
            source_dir = dir + '/' + fn + '/' + type + '/' + '2009*' 

            new_dir = target_dir + '/' + type + '/'

            print("new_dir:", new_dir)

            cp_cmd = 'cp -r ' + source_dir + ' ' + new_dir

            print(cp_cmd)

            # if count > 5:
            #     exit(0)
            
            try:
                if copy_flag:
                    os.system(cp_cmd)
            except Exception as e:
                print(e)
                pass

            count = count +1

        else:
            print('File fn:', fn)

    print('count:', count)


target_dir = '/home/ascc/Desktop/test_set/'
dir = '/home/ascc/Desktop/adl_0819v1/activity_data/'
copy_ascc_act_dir(dir,  target_dir, type = 'Motion')
copy_ascc_act_dir(dir,  target_dir, type = 'Motion', copy_flag=True)
copy_ascc_act_dir(dir,  target_dir, type = 'Audio', copy_flag=True)
copy_ascc_act_dir(dir,  target_dir, type = 'Image', copy_flag=True)


exit(0)

#image_rotate(IMAGE_TEST)

# count = get_file_count_of_dir(IMAGE_DIR)
# count = get_rotate_file_of_dir(IMAGE_DIR, IMAGE_SAVE_DIR)
# count = get_rotate_file_from_dir(IMAGE_DIR)

image_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Reading/'
image_dest_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Reading_pad'
image_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Morning_Med/'
image_dest_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Morning_Med_pad'
image_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Meditate/'
image_dest_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Meditate_pad'
image_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/LeavingHome/'
image_dest_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/LeavingHome_pad'
image_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Eve_Med/'
image_dest_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Eve_Med_pad'
image_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Dining_room_activity/'
image_dest_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Dining_room_activity_pad'
image_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Desk_activity/'
image_dest_dir = '/home/ascc/Desktop/data_activity_0217/data_0217/Res_activity/Desk_activity_pad'

image_dir = '/home/ascc/Desktop/image_living_bedroom/new_data/bed_rotate/'
image_dest_dir = '/home/ascc/Desktop/image_living_bedroom/new_data/bed_res/'

count = generate_copy_image_file_from_dir(image_dir, image_dest_dir)

print("Res file count:", count)
