"""Trains a DQN/DDQN to solve CartPole-v0 problem


"""
#import ground_truth_dict_dataset0819


import json
import math


import matplotlib.pyplot as plt


def plotone(rewards, fig = "test.png", xlabel = "Episodes", ylabel="Rewards"):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend()
    plt.savefig(fig)

    #plt.show()
    plt.clf()

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








total_wmu_cam_trigger_times = [5530, 3964, 2752, 1925, 1313, 892, 605, 440, 313, 226, 174, 133, 101, 83, 81, 93, 82, 73]

total_wmu_mic_trigger_times = [3717, 2467, 1777, 1229, 822, 565, 368, 261, 171, 105, 93, 45, 39, 33, 138, 292, 21, 12]

total_wmu_trigger_times = [1867, 1244, 940, 619, 412, 280, 199, 148, 94, 60, 63, 31, 31, 32, 28, 30, 18, 9]

total_robot_mic_trigger_times = [5802, 5949, 4325, 4678, 3799, 3663, 2371, 7088, 6508, 6563, 1901, 914, 1040, 329, 364, 745, 275, 267]

total_robot_cam_trigger_times = [3742, 2670, 1960, 1374, 974, 704, 542, 415, 330, 299, 255, 249, 180, 196, 195, 189, 211, 207]
total_robot_trigger_times = [1924, 1380, 1062, 793, 589, 450, 366, 285, 245, 250, 223, 226, 178, 191, 194, 182, 209, 204]

total_privacy_times = [596, 448, 334, 204, 170, 128, 129, 98, 105, 82, 90, 77, 70, 77, 75, 69, 79, 66]


scores = [-5748.73999999987, -4065.63999999983, -2722.4799999999263, -1818.1599999999328, -1142.4999999999498, -707.2199999999509, -402.4399999999704, -330.63999999996065, -193.74000000003394, -82.56000000000225, 31.760000000000645, 106.89999999999955, 153.62000000000168, 172.58000000000015, 166.64000000000007, -9.459999999999685, 180.91999999999985, 201.3799999999998]

default_x_ticks = range(len(scores))

print('len(scores):', len(scores))
#plotone(scores, fig = "dqn_accumulated_rewards.png", xlabel = "Episodes", ylabel="Accumulated Rewards")

plt.figure(figsize=(20,5))
plt.rcParams.update({'font.size': 23})
plt.plot(default_x_ticks, scores)
plt.xticks(default_x_ticks)
# plt.xlabel('Episodes')
# plt.ylabel('Accumulated Rewards')
# plt.legend()
plt.savefig('ddqn_accumulated_rewards.png')
plt.clf()



print('len(wmu times):', len(total_wmu_cam_trigger_times))

plt.figure(figsize=(20,5))
plt.rcParams.update({'font.size': 23})
plt.plot(total_wmu_cam_trigger_times, label='wmu_cam_trigger_times')
plt.plot(total_wmu_mic_trigger_times, label='wmu_mic_trigger_times')
#plt.plot(total_wmu_trigger_times, label='total_wmu_mic_cam_trigger_times')
plt.plot(total_robot_mic_trigger_times, label='total_robot_mic_trigger_times')
plt.plot(total_robot_cam_trigger_times, label='total_robot_cam_trigger_times')
#plt.plot(total_robot_trigger_times, label='total_robot_mic_cam_trigger_times')
plt.plot(total_privacy_times, label='privacy_violation_occurring_times')



plt.xticks(default_x_ticks)

# plt.xlabel("Episodes")
# plt.ylabel("Times")
plt.legend()
plt.savefig('ddqn_multi_reward.png')

# plt.show()
plt.clf()

exit(0)

