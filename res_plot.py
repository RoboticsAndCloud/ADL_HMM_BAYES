# Adding a Y-Axis Label to the Secondary Y-Axis in Matplotlib
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
 
def plot(rewards1, rewards2):
    plt.figure(figsize=(20,5))
    plt.plot(rewards1, label='rewards1')
    plt.plot(rewards2, label='rewards2')

    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig('multi_reward.png')

    plt.show()
    plt.clf()


Detection_Ratio = [0.89, 0.85, 0.70, 0.68, 0.85, 0.91, 0.93]
Trigger_Times = [1537, 805, 412, 277, 276, 140, 78]
Extra_Energy = [1538.92, 806, 412.56, 277.35, 167.67, 140.17, 78.1]



# creating data for plot
# data arrangement between 0 and 50
# with the difference of 2
# x-axis
x = np.arange(0, len(Detection_Ratio))

# Custome x labels
xlabels = ['0.5 Mins(Period)', '1 Min(Period)', '2 Mins(Period)', '3 Mins(Period)', 'Proposed_vision+motion', 'Proposed_vision+motion+sound', 'Proposed_real_time_test']
xlabels = ['0.5 Mins(Period)', '1 Min(Period)', '2 Mins(Period)', '3 Mins(Period)', 'Proposed_vision+motion', 'Proposed_vision+motion+sound', 'Proposed_real_time_test']

#plt.xticks(x, xlabels, rotation='vertical')

# y-axis values
y1 = Trigger_Times
y11 = Extra_Energy
 
# secondary y-axis values
y2 = Detection_Ratio
 
# plotting figures by creating axes object
# using subplots() function
fig, ax = plt.subplots(figsize = (10, 5))
plt.title('Results between Periodic Methods and the Proposed Method', fontsize= 'x-large')
 
# using the twinx() for creating another
# axes object for secondary y-Axis
ax2 = ax.twinx()
l_trigger_time, = ax.plot(x, y1, color = 'g', label='Trigger_Times')
l_energy, = ax.plot(x, y11, color = 'r', label='Extra_Energy')

l_detection_ration, = ax2.plot(x, y2, color = 'b', label='Detection_Ratio')
 
# giving labels to the axises
#ax.set_xlabel('x-axis', color = 'r')
ax.set_xticks(x)
ax.set_xticklabels(xlabels,rotation = 30, fontsize = 'x-large') # 设置刻度标签

ax.set_ylabel('Trigger Times & Energy Cost', fontsize= 'x-large', color = 'g')
#ax.legend(loc='lower right')
 
# secondary y-axis label
ax2.set_ylabel('Detection Ratio', fontsize= 'x-large', color = 'b')
#ax2.legend(loc='lower right')
 
# defining display layout 
plt.tight_layout()
 
#plt.legend()
plt.legend(handles=[l_trigger_time, l_energy, l_detection_ration],labels =["Trigger Times", "Extra Energy", "Detection Ratio"])

# show plot
plt.savefig('Multi_y_axis.png')

#plt.legend(handles=[l1,l2],labels=['up','down'],loc='best')

plt.show()
plt.clf()

