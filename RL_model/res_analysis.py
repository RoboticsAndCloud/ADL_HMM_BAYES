"""Trains a DQN/DDQN to solve CartPole-v0 problem


"""



import matplotlib.pyplot as plt


def plotone(rewards, fig = "test.png"):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
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


#scores =[-1011.2999999998245, -1013.0599999998111,-1048.499999999825, -854.9999999998544, -129.20000000006905, 509.6799999999743, 175.85999999994436, -243.05999999998932, -655.4000000000583, -71.8600000000444, 1548.6599999999617, 1322.9000000000174, -56.980000000034146, 2155.919999999853, 5303.299999999906, 4003.83999999987, 8771.080000000231, 4365.779999999948,5146.059999999576, 4580.040000000401, 2854.8799999992066, 6064.599999999398, 3974.8599999996927, 3571.7399999989048, 3542.6599999989403, 3878.9799999986076]
scores = [-4985.4999999999045, -4682.699999999822, -4914.200000000065, -4200.400000000005, -3375.3000000000084, -4100.999999999398,-3680.1000000000254, -1382.4000000001197, -1881.0000000002944, 3022.500000000364, 3921.00000000098, 4303.000000001262, 4157.600000001274, 4418.400000001332]
print(len(scores))
plotone(scores, fig='episode_reward.png')

wmu_cam_times = [9320, 8963, 8106, 8502, 6250, 8806,7308, 3894, 7072, 2485,  1494, 1391, 1709, 1280]
print(len(scores))
plotone(scores, fig='episode_wmu_cam_times.png')

exit(0)

