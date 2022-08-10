#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################
import logging
import time

import cv2
import argparse
import numpy as np
import os
import utils

# import image_rotate

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def check_and_wait_for_food_images_by_dir(dir):
    image_count = utils.get_dirfile_count_of_dir(dir)

    print("Check_and_wait_for_food_images...")
    logging.info("Image image_count: %d", image_count)

    while(True):
        new_count = utils.get_dirfile_count_of_dir(dir)

        if new_count == image_count:
            continue
        else:
            logging.info("Image new_count: %d", new_count)
            time.sleep(1)
            image_count = new_count
            break

    return image_count

def check_and_wait_for_food_images(dir):
    image_count = utils.get_file_count_of_dir(dir)

    print("Check_and_wait_for_food_images...")
    logging.info("Image image_count: %d", image_count)

    while(True):
        new_count = utils.get_file_count_of_dir(dir)

        if new_count == image_count:
            continue
        else:
            logging.info("Image new_count: %d", new_count)
            time.sleep(1)
            image_count = new_count
            break

    return image_count

def check_labels(class_ids):
    """
    check the image is blurred  or the image could used for detecting pizza
    :param img:
    :param class_ids:
    :return: object_flat,blur_flag
    """

    dest_label = 'apple'

    flag = False
    blur_flag = True
    for i in range(len(class_ids)):
        label = str(classes[class_ids[i]])
        if label == dest_label:
            flag = True
            blur_flag = False
            break
        if label == 'dining table':
            blur_flag = False

    if blur_flag == True:
        blur_str = "Image is blurred, Please adjust the position and Retake images"
        print(blur_str)
        time.sleep(1.5)
        utils.TTSTool.tts(blur_str)
        return flag, blur_flag

    if flag == True:
        detect_str = "Your are eating " + dest_label
        print(detect_str)
        time.sleep(1.8)
        utils.TTSTool.tts(detect_str)

    else:
        adjust_and_retake_str = "Please adjust the position and Retake images"
        print(adjust_and_retake_str)
        time.sleep(2)
        utils.TTSTool.tts(adjust_and_retake_str)

    return flag, blur_flag


def check_object_location(x, x_plus_w, image_width, class_id):
    # check the object location in the image
    label = str(classes[class_id])

    if label != 'apple' and label != 'pizza':
        return

    obj_width = x + x_plus_w
    if obj_width < image_width/3:
        print("The object %s is on the left, Please adjust the position and Retake images", label)
    elif  image_width/3 <= obj_width:
        print("The object %s is on the right, Please adjust the position and Retake images", label)

    # if detect nothing, maybe blur, retake the image



def draw_prediction_length(img, class_id, confidence, x, y, w, h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    x_plus_w = round(x + w)
    y_plus_h = round(y + h)
    ## Distance Meaasurement for each bounding box


    ## item() is used to retrieve the value from the tensor

    distance = (2 * 3.14 * 180) / (w + h * 360) * 1000 + 3 ### Distance measuring in Inch

    feedback = ("{}".format(label+ " " +":"+" at {} ".format(round(distance))+"Inches"))
    print(feedback)

    x = round(x)
    y = round(y)
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    text = label + ":" + feedback


    cv2.putText(img, feedback, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    if label == 'refrigerator' or label == 'oven' or label == 'chair' or label == 'spoon':
        return

    print(label)

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# todo read files from dir and generate the result image

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
                detection_file = rotate_dir + '/' + fn.strip('.jpg') + '_detection.jpg'

                object_detection(image_file, detection_file)
                # try:
                #     image_rotate2(image_file, rotate_file)
                # except Exception as e:
                #     print(e)
                #     pass


                count = count + 1
        else:
            print('fn:', fn)
    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count


def get_rotate_file_of_dir_by_last_photo(dir, rotate_dir, image_count):
    path = dir
    count = 0
    prefix = ''
    dir_list = os.listdir(path)
    dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(path, x)))
    print(dir_list)


    for fn in dir_list:
        if os.path.isfile(dir + '/' + fn):
            if prefix != '':
                if prefix in fn:
                    count = count + 1
            else:
                print(fn)
                image_file = dir + '/' + fn
                detection_file = rotate_dir + '/' + fn.strip('.jpg') + '_detection.jpg'

                # try:
                #     image_rotate2(image_file, rotate_file)
                # except Exception as e:
                #     print(e)
                #     pass


                count = count + 1
                if count == image_count:
                    object_detection(image_file, detection_file)
                    break

        else:
            print('fn:', fn)
    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count


def food_detection_from_dir_by_10_images(dir, rotate_dir):
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

                # just detect rotated images, ignore original
                if fn.find('rotate') == -1:
                    continue

                image_file = dir + '/' + fn
                detection_file = rotate_dir + '/' + fn.strip('.jpg') + '_detection.jpg'
                count = count + 1

                if count == 5:
                    object_detection(image_file, detection_file)

                # try:
                #     image_rotate2(image_file, rotate_file)
                # except Exception as e:
                #     print(e)
                #     pass


        else:
            print('fn:', fn)
    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count

def food_detectiong_dir_by_last_photo_from_dir(dir, rotate_dir, dir_count):
    path = dir
    count = 0
    prefix = ''
    dir_list = os.listdir(path)
    dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(path, x)))
    print(dir_list)



    for fn in dir_list:
        if os.path.isdir(dir + '/' + fn):
            if prefix != '':
                if prefix in fn:
                    count = count + 1
            else:
                print(fn)
                image_file_dir = dir + '/' + fn
                detection_file_dir = rotate_dir + '/' + fn + '_rotate'

                try:
                    if os.path.exists(detection_file_dir) == False:
                        os.mkdir(detection_file_dir)
                except Exception as e:
                    print(e)
                    pass

                image_rotate.get_rotate_file_of_dir(image_file_dir, detection_file_dir)

                # try:
                #     image_rotate2(image_file, rotate_file)
                # except Exception as e:
                #     print(e)
                #     pass


                count = count + 1
                if count == dir_count:

                    food_detection_from_dir_by_10_images(detection_file_dir, detection_file_dir)
                    break

        else:
            print('fn:', fn)
    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count


def object_detection(image_str, image_save_path):

    print(args.image)
    image = cv2.imread(image_str)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # classes = None
    #
    # with open(args.classes, 'r') as f:
    #     classes = [line.strip() for line in f.readlines()]

    # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    check_labels(class_ids)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        #draw_prediction_length(image, class_ids[i], confidences[i], (x), (y), (w), (h))

        # check_object_location(x, round(x+w), Width, class_ids[i])

        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    print("Running")

    cv2.imshow("object detection", image)
    # cv2.waitKey()

    cv2.imwrite(image_save_path, image)
    cv2.destroyAllWindows()

def run():
    image_dir = '/home/ascc/asccbot_v3/wearable_device_new_design/server/images/'
    res_dir = '/home/ascc/asccbot_v3/wearable_device_new_design/server/images_food_res/'

    print(image_dir)
    print('Running...')

    # while(True):

    #     # be careful, the 10 consective images could be in one folder
    #     # image_count = check_and_wait_for_food_images(image_dir)
    #     image_count = check_and_wait_for_food_images_by_dir(image_dir)

    #     print("image_count ", image_count)
    #     time.sleep(10)

    #     food_detectiong_dir_by_last_photo_from_dir(image_dir, res_dir, image_count)

    #     #get_rotate_file_of_dir_by_last_photo(image_dir, res_dir, image_count)

    #     # get_rotate_file_of_dir(image_dir, res_dir)


test_img = '/home/ascc/Desktop/test/read.jpg'
object_detection(test_img, test_img+'_detection')
#image_dir = '/home/ascc/lf_workspace/ObjectDetection/object-detection-opencv/dataset/Apple_rotate_all/'
#res_dir = '/home/ascc/lf_workspace/ObjectDetection/object-detection-opencv/dataset/Apple_res/'

# image_dir = '/home/ascc/lf_workspace/ObjectDetection/object-detection-opencv/dataset/' + 'Pizza_all'
# res_dir = '/home/ascc/lf_workspace/ObjectDetection/object-detection-opencv/dataset/' + 'Pizza_res'

# image_dir = '/home/ascc/lf_workspace/ObjectDetection/object-detection-opencv/dataset/' + 'Mix_all'
# res_dir = '/home/ascc/lf_workspace/ObjectDetection/object-detection-opencv/dataset/' + 'Mix_res'

# image_dir = '/home/ascc/lf_workspace/ObjectDetection/object-detection-opencv/dataset/' + 'Banana_all'
# res_dir = '/home/ascc/lf_workspace/ObjectDetection/object-detection-opencv/dataset/' + 'Banana_res'

# image_dir = '/home/ascc/asccbot_v3/wearable_device_new_design/server/images/'
# res_dir = '/home/ascc/asccbot_v3/wearable_device_new_design/server/images_food_res/'
#
# import log
# log.init_log("./log/my_program")  # ./log/my_program.log./log/my_program.log.wf7
# logging.info("Hello World!!!")
#
# print(image_dir)
#
# check_and_wait_for_food_images(image_dir)
#
# get_rotate_file_of_dir(image_dir, res_dir)

if __name__ == "__main__":
    import log

    log.init_log("./log/my_program")  # ./log/my_program.log./log/my_program.log.wf7
    logging.info("Hello World!!!")

    run()
