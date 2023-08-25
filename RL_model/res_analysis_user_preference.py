"""Trains a DQN/DDQN to solve CartPole-v0 problem


"""



import matplotlib.pyplot as plt


def plotone(rewards, fig = "test.png", x = "Episodes", y = "Accumulated Rewards"):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.xlabel(x)
    plt.ylabel(y)
    #plt.xlabel("Episode")
    #plt.ylabel("Rewards")
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

# basic, changes, retrain
# wmu_cam = [122, 119, 199]
# robot_cam = [226, 283, 170]
# privacy = [0, 45, 0]
# day = [1, 2, 3]

# # without adaptation, just one basic model
# wmu_cam = [122, 98, 90, 115, 100, 87, 123]
# robot_cam = [226, 167, 188, 198, 219, 195, 229]
# privacy = [0, 0, 0, 40, 45, 38, 44]
# day = [1, 2, 3, 4, 5, 6, 7]



# wmu_cam1 = [122, 122, 122, 121, 73, 73, 73]
# robot_cam1 = [226, 226, 226, 223, 277, 277, 277]
# privacy1 = [0, 0, 0, 0, 0, 0, 0]

# plt.figure(figsize=(20,5))
# plt.plot(day, wmu_cam, 'b-*', label='wmu_cam')
# plt.plot(day, robot_cam, 'y-*', label='robot_cam')
# plt.plot(day, privacy, 'r-*', label='privacy_violation')


# # plt.plot(day, wmu_cam1, 'b-.d', label='wmu_cam')
# # plt.plot(day, robot_cam1, 'y-.d', label='robot_cam')
# # plt.plot(day, privacy1, 'r-.d', label='privacy_violation')


# plt.xlim(1, 7.1)
# plt.xlabel("Day")
# plt.ylabel("Sensor Trigger Times")
# plt.legend()
# plt.savefig('user1.png')

# plt.show()
# plt.clf()

# exit(0)


# # basic, changes, retrain
# wmu_cam = [122, 119, 199]
# robot_cam = [226, 283, 170]
# privacy = [0, 45, 0]
# day = [1, 2, 3, 4, 5, 6, 7]



#  adaptation, just one basic model
wmu_cam = [122, 98, 90, 115, 188, 166, 200]
robot_cam = [226, 167, 188, 198, 170, 142, 174]
privacy = [0, 0, 0, 40, 1, 1, 2]
day = [1, 2, 3, 4, 5, 6, 7]

plt.figure(figsize=(20,5))
plt.plot(day, wmu_cam, 'b-*', label='wmu_cam')
plt.plot(day, robot_cam, 'y-*', label='robot_cam')
plt.plot(day, privacy, 'r-*', label='privacy_violation')


# plt.plot(day, wmu_cam1, 'b-.d', label='wmu_cam')
# plt.plot(day, robot_cam1, 'y-.d', label='robot_cam')
# plt.plot(day, privacy1, 'r-.d', label='privacy_violation')


plt.xlim(1, 7.1)
plt.xlabel("Day")
plt.ylabel("Sensor Trigger Times")
plt.legend()
plt.savefig('user2.png')

plt.show()
plt.clf()

exit(0)



wmu_cam1 = [122, 121, 73]
robot_cam1 = [226, 223, 277]
privacy1 = [0, 0, 0]



