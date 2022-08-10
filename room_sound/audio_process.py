import os

def copy_file(dir, target_dir):
    path = dir
    count = 0
    prefix = ''
    for fn in os.listdir(path):
        if os.path.isdir(dir + '/' + fn):
            print(fn)
            s_file = dir + '/' + fn +'/recorded.wav'
            d_file = target_dir + '/' + fn +'_recorded.wav'
            cp_cmd = 'cp ' + s_file + ' ' + d_file
            os.system(cp_cmd)
            print(cp_cmd)
            count = count + 1
        # else:
        #     print('fn:', fn)
    # count = sum([len(files) for root, dirs, files in os.walk(dir)])
    return count

#ffmpeg -ss 30 -i input.wmv -c copy -t 10 output.wmv
def audio_file_truncate(source_dir, target_dir):
    path = source_dir
    count = 0
    
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for fn in os.listdir(path):
        if os.path.isfile(path + '/' + fn):
            print(fn)
            input = source_dir + '/' + fn
            output = target_dir + '/' + fn
            
            #ffmpeg -ss 0 -i input -c copy -t 1 (output + '0').wmv
            cmd = 'ffmpeg -ss 0 -i ' + input + ' -c copy -t 1 ' + output + '_0' + '.wav'
            print(cmd)
            os.system(cmd)

            #ffmpeg -ss 1 -i input -c copy -t 1 (output + '1').wmv
            cmd = 'ffmpeg -ss 1 -i ' + input + ' -c copy -t 1 ' + output + '_1' + '.wav'
            print(cmd)
            os.system(cmd)

            #ffmpeg -ss 2 -i input -c copy -t 1 (output + '2').wmv
            cmd = 'ffmpeg -ss 2 -i ' + input + ' -c copy -t 1 ' + output + '_2' + '.wav'
            print(cmd)
            os.system(cmd)


            #ffmpeg -ss 3 -i input -c copy -t 1 (output + '3').wmv
            cmd = 'ffmpeg -ss 3 -i ' + input + ' -c copy -t 1 ' + output + '_3' + '.wav'
            print(cmd)
            os.system(cmd)

            #ffmpeg -ss 4 -i input -c copy -t 1 (output + '4').wmv
            cmd = 'ffmpeg -ss 4 -i ' + input + ' -c copy -t 1 ' + output + '_4' + '.wav'
            print(cmd)
            os.system(cmd)


            #os.system(cmd)
            count = count + 1

    return count


def audio_file_truncate_by_dir(dir):
    path = dir
    count = 0
    prefix = ''
    for fn in os.listdir(path):
        if os.path.isdir(dir + '/' + fn):
            source_dir = dir + '/' + fn
            target_dir = source_dir.replace('ascc_activity', 'ascc_activity_1second')
            print(source_dir)
            print(target_dir)

            cnt = audio_file_truncate(source_dir, target_dir)
            print('Count:', cnt)

            count = count + cnt

    return count



if __name__ == '__main__':
    print("Running")

    source_dir = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/test'
    target_dir = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/test_result'


    source_dir = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/ascc_activity/door_open_closed'
    target_dir = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/ascc_activity_1second/door_open_closed'

    source_dir = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/ascc_activity/eating'
    target_dir = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/ascc_activity_1second/eating'

    source_dir = '/home/ascc/Desktop/Sample_set/whand/'
    target_dir = '/home/ascc/Desktop/Sample_set/whand_1sec/'

    
    cnt = audio_file_truncate(source_dir, target_dir)
    print('Count:', cnt)
    
    # audio_file_truncate_by_dir('/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/ascc_activity/')   


# dir = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/senior_design/Server/data/audio/'
# target = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/rpi/'
#cnt = copy_file(dir, target)
#print("count:", cnt)







