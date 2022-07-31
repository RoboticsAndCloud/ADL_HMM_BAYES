"""

Write motion info file /home/pi/Desktop/data/motion/20220729101522//motion.txt
Write motion info file /home/pi/Desktop/data/motion/20220729101732//motion.txt
Write motion info file /home/pi/Desktop/data/motion/20220729101937//motion.txt
Write motion info file /home/pi/Desktop/data/motion/20220729102128//motion.txt

motion.txt

-1.1375714      -9.022118       -0.6668522
-1.1375714      -9.022118       -0.6668522
-1.0591182      -9.022118       -0.6668522
-1.1375714      -8.9436648      -0.6668522


33,Jogging,49105962326000,-0.6946377,12.680544,0.50395286;
33,Jogging,49106062271000,5.012288,11.264028,0.95342433;
33,Jogging,49106112167000,4.903325,10.882658,-0.08172209;
33,Jogging,49106222305000,-0.61291564,18.496431,3.0237172;
33,Jogging,49106332290000,-1.1849703,12.108489,7.205164;


columns = ['user', 'activity', 'time', 'x', 'y', 'z']

"""

file_dict = {
    'sitting': ['20220729101522', '20220729101732', '20220729101937', '20220729102128']
}

