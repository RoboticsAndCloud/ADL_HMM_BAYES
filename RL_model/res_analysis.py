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

    plt.show()
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


scores =[-1048.499999999825, -854.9999999998544, -129.20000000006905, 509.6799999999743, 175.85999999994436, -243.05999999998932, -655.4000000000583, -71.8600000000444, 1548.6599999999617, 1322.9000000000174, -56.980000000034146, 2155.919999999853, 5303.299999999906, 4003.83999999987, 8771.080000000231, 4365.779999999948,5146.059999999576, 4580.040000000401, 2854.8799999992066, 6064.599999999398]
plotone(scores, fig='episode_reward.png')
exit(0)

