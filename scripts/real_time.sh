gnome-terminal -x bash -c "cd /home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES && source  ~/LF_Workspace/venv3.7_energy/bin/activate && python adl_web_mq_server.py"
sleep 10
echo 'Start adl_web_mq_server'


gnome-terminal -x bash -c "cd /home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES && source  ~/LF_Workspace/venv3.7_energy/bin/activate && python adl_server_main_node.py"
sleep 10
echo 'Start adl_server_main_node'



gnome-terminal -x bash -c "cd /home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier && source  ~/LF_Workspace/venv3.7_energy/bin/activate && python ascc_robot_room_activity_test.py"
sleep 3
echo 'Start robot image recognition'


gnome-terminal -x bash -c "cd /home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier && source  ~/LF_Workspace/venv3.7_energy/bin/activate && python ascc_room_activity_test.py"
sleep 3
echo 'Start wmu image recognition'


gnome-terminal -x bash -c "cd /home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/room_sound && source  ~/LF_Workspace/venv3.7_energy/bin/activate && python ascc_room_sound_activity_test.py"
sleep 3
echo 'Start sound event recognition'


gnome-terminal -x bash -c "cd /home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/room_motion_activity && source  ~/LF_Workspace/venv3.7_energy/bin/activate && python ascc_room_motion_activity_test.py"
sleep 3
echo 'Start motion recognition'


gnome-terminal -x bash -c "cd /home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/room_classifier/ImageProcess && source  ~/LF_Workspace/venv3.7_energy/bin/activate && python food_yolo_opencv.py --config yolov3.cfg --weights  yolov3.weights --class yolov3.txt"
sleep 3
echo 'Start object recognition'



# Env requirements:
# source  ~/LF_Workspace/venv3.7_energy/bin/activate


# (venv3.7_energy) (base) ascc@ascc-XPS-8940:~/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES$ python adl_web_mq_server.py


# (venv3.7_energy) (base) ascc@ascc-XPS-8940:~/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES$ python adl_server_main_node.py

# (venv3.7_energy) (base) ascc@ascc-XPS-8940:~/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier$ python ascc_robot_room_activity_test.py 


# Image Reconition for WMU images
# (venv3.7_energy) (base) ascc@ascc-XPS-8940:~/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier$ python ascc_room_activity_test.py 




# Test WMU
# (venv3.7_energy) (base) ascc@ascc-XPS-8940:~/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/adl_wmu$ python test_adl_server_main_wmu.py 

# Robot:
# python2.7 adl_server_main_companion_robot.py 


# RL model:
# source  ~/LF_Workspace/venv3.8_rl_pytorch/bin/activate
# (venv3.8_rl_pytorch) (base) ascc@ascc-XPS-8940:~/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/RL_model$ python rl_real_test_dqn.py