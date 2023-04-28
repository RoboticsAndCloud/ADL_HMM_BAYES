#!/bin/bash
sudo chmod 777 /dev/ttyACM0

gnome-terminal -x bash -c "roscore"

sleep 5


gnome-terminal -x bash -c "rosrun rosserial_python serial_node.py '/dev/ttyACM0'"
gnome-terminal -x bash -c "rosrun voice_interface sent_angle_message_02.py"
#gnome-terminal -x bash -c "export PYTHONPATH=/usr/local/lib/python3.7/dist-packages:$PYTHONPATH && source ~/szd-python3-env/bin/activate && rosrun voice_interface RobotWMU_head_control.py"
gnome-terminal -x bash -c "export PYTHONPATH=/usr/local/lib/python3.7/dist-packages:$PYTHONPATH && source ~/szd-python3-env/bin/activate && rosrun voice_interface telegram_chat_head_control.py"
