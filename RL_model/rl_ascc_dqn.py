"""Trains a DQN/DDQN to solve CartPole-v0 problem
https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras.git


"""

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
from collections import deque
import numpy as np
import random
import argparse
import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt

from tensorflow.keras import backend as K

#config = tf.compat.v1.ConfigProto(log_device_placement=True)
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#K.set_session(session)

MODEL_SAVED_PATH = 'ascc_rl_dqn-saved-model'

def plot(rewards, figure = 'ascc_rl_reward.png'):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    # plt.legend()
    plt.savefig(figure)

    # plt.show()
    plt.clf()


class DQNAgent:
    def __init__(self,
                 state_space, 
                 action_space, 
                 episodes=500, epsilon = 1.0, memory_size = 51200):
        """DQN Agent on CartPole-v0 environment

        Arguments:
            state_space (tensor): state space
            action_space (tensor): action space
            episodes (int): number of episodes to train
        """
        self.action_space = action_space

        # experience buffer
        self.memory = []
        self.memory_transition = []
        self.memory_size = memory_size

        # discount rate
        self.gamma = 0.9

        # initially 90% exploration, 10% exploitation
        self.epsilon = epsilon
        # iteratively applying decay til 
        # 10% exploration/90% exploitation
        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** \
                             (1. / float(episodes))

        # Q Network weights filename
        self.weights_file = 'dqn_cartpole.h5'
        # Q Network for training
        n_inputs = state_space
        n_outputs = len(action_space)
        self.q_model = self.build_model(n_inputs, n_outputs)
        self.q_model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        #self.q_model.compile(loss='mse', optimizer=Adam(learning_rate=0.005))
        # target Q Network
        self.target_q_model = self.build_model(n_inputs, n_outputs)
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0

    
    def build_model22(self, n_inputs, n_outputs):
        """Q Network is 256-256-256 MLP

        Arguments:
            n_inputs (int): input dim
            n_outputs (int): output dim

        Return:
            q_model (Model): DQN
        """
        inputs = Input(shape=(n_inputs, ), name='state')
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(n_outputs,
                  activation='linear', 
                  name='action')(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model

    def build_model(self, n_inputs, n_outputs):
        """Q Network is 256-256-256 MLP

        Arguments:
            n_inputs (int): input dim
            n_outputs (int): output dim

        Return:
            q_model (Model): DQN
        """
        inputs = Input(shape=(n_inputs, ), name='state')
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        #x = Dense(64, activation='relu')(x)
        #x = Dense(64, activation='relu')(x)
      #  x = Dense(64, activation='relu')(x)
      #  x = Dense(64, activation='relu')(x)
        #x = Dropout(0.2)(x)
        x = Dense(n_outputs,
                  activation='linear', 
                  name='action')(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model


    def save_weights(self):
        """save Q Network params to a file"""
        self.q_model.save_weights(self.weights_file)
        self.q_model.save(MODEL_SAVED_PATH)

    def load_weights(self):
        """save Q Network params to a file"""
        # self.q_model.load_weights(self.weights_file)
        from keras.models import load_model
        self.q_model = load_model(MODEL_SAVED_PATH)
        self.target_q_model.set_weights(self.q_model.get_weights())
        self.q_model.summary()
        print('load weights')

    def update_weights(self):
        """copy trained Q Network params to target Q Network"""
        self.target_q_model.set_weights(self.q_model.get_weights())


    def act(self, state):
        """eps-greedy policy
        Return:
            action (tensor): action to execute
        """
        if np.random.rand() < self.epsilon:
            # explore - do random action
           # if np.random.rand() < 0.5:
           #     return random.sample([3,1], 1)[0] # let robot_fusion action get more chance
            return random.sample(self.action_space,1)[0]

        # exploit
        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        print("q_values:", q_values)
        action = np.argmax(q_values[0])
        return action
    
    def update_replay_memory(self):

        #self.memory = random.sample(self.memory, int(len(self.memory)/20))

        self.memory.clear()

        self.memory = []
        


    def remember(self, state, action, reward, next_state, done):
        """store experiences in the replay buffer
        Arguments:
            state (tensor): env state
            action (tensor): agent action
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        """
        item = (state, action, reward, next_state, done)
        self.memory.append(item)

        if len(self.memory) > self.memory_size:
            index = int(self.memory_size/2)
            self.memory = self.memory[index:]

        # self.memory = [] # ring buffer
        # # replace the old memory with new memory
        # index = self.memory_counter % self.memory_size

    def remember_transition(self, state, action, reward, next_state, done):
        """store experiences in the replay buffer
        Arguments:
            state (tensor): env state
            action (tensor): agent action
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        """
        item = (state, action, reward, next_state, done)
        self.memory_transition.append(item)

        if len(self.memory_transition) > self.memory_size:
            index = int(self.memory_size/2)
            self.memory_transition = self.memory_transition[index:]


    def get_target_q_value2(self, next_state, reward):
        """compute Q_max
           Use of target Q Network solves the 
            non-stationarity problem
        Arguments:
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        Return:
            q_value (float): max Q-value computed
        """
        # max Q value among next state's actions
        # DQN chooses the max Q value among next actions
        # selection and evaluation of action is 
        # on the target Q Network
        # Q_max = max_a' Q_target(s', a')
        tensor_state=tf.convert_to_tensor(next_state)
        #q_values = self.q_model(tensor_state).numpy()
        q_value = np.amax(\
                     self.target_q_model(tensor_state).numpy()[0])

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value

    def get_target_q_value(self, next_state, reward):
        """compute Q_max
           Use of target Q Network solves the 
            non-stationarity problem
        Arguments:
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        Return:
            q_value (float): max Q-value computed
        """
        # max Q value among next state's actions
        # DQN chooses the max Q value among next actions
        # selection and evaluation of action is 
        # on the target Q Network
        # Q_max = max_a' Q_target(s', a')
        q_value = np.amax(\
                     self.target_q_model.predict(next_state)[0])

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value


    def replay2(self, batch_size, transition=False):
        """experience replay addresses the correlation issue 
            between samples
        Arguments:
            batch_size (int): replay buffer batch 
                sample size
        """
        # sars = state, action, reward, state' (next_state)
        print('len memory:', len(self.memory), ' ', len(self.memory_transition))
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        if transition:
            sars_batch = random.sample(self.memory_transition, batch_size)


        # fixme: for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for state, action, reward, next_state, done in sars_batch:

            tensor_state=tf.convert_to_tensor(state)
            q_values = self.q_model(tensor_state).numpy()

            # policy prediction for a given state
            #q_values = self.q_model.predict(state)
            
            # get Q_max
            q_value = self.get_target_q_value2(next_state, reward)

            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])
            #K.clear_session()

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=0)

        # update exploration-exploitation probability
        self.update_epsilon()

        # copy new params on old target after 
        # every 10 or 5 training updates
        if self.replay_counter % 10 == 0:
            self.update_weights()

        self.replay_counter += 1

    def replay(self, batch_size):
        """experience replay addresses the correlation issue 
            between samples
        Arguments:
            batch_size (int): replay buffer batch 
                sample size
        """
        # sars = state, action, reward, state' (next_state)
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        # fixme: for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for state, action, reward, next_state, done in sars_batch:

            #tensor_state=tf.convert_to_tensor(state)
            #q_values = self.q_model.predict(tensor_state)

            # policy prediction for a given state
            q_values = self.q_model.predict(state)
            
            # get Q_max
            q_value = self.get_target_q_value(next_state, reward)

            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=0)

        # update exploration-exploitation probability
        self.update_epsilon()

        # copy new params on old target after 
        # every 10 training updates
        if self.replay_counter % 5 == 0:
            self.update_weights()

        self.replay_counter += 1



    
    def update_epsilon(self):
        """decrease the exploration, increase exploitation"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        

class DDQNAgent(DQNAgent):
    def __init__(self,
                 state_space, 
                 action_space, 
                 episodes=500):
        super().__init__(state_space, 
                         action_space, 
                         episodes)
        """DDQN Agent on CartPole-v0 environment

        Arguments:
            state_space (tensor): state space
            action_space (tensor): action space
            episodes (int): number of episodes to train
        """

        # Q Network weights filename
        self.weights_file = 'ddqn_cartpole.h5'
        print("-------------DDQN------------")

    def get_target_q_value(self, next_state, reward):
        """compute Q_max
           Use of target Q Network solves the 
            non-stationarity problem
        Arguments:
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        Returns:
            q_value (float): max Q-value computed
        """
        # max Q value among next state's actions
        # DDQN
        # current Q Network selects the action
        # a'_max = argmax_a' Q(s', a')
        action = np.argmax(self.q_model.predict(next_state)[0])
        # target Q Network evaluates the action
        # Q_max = Q_target(s', a'_max)
        q_value = self.target_q_model.predict(\
                                      next_state)[0][action]

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value
    


def test_model():

    # the number of trials without falling over
    win_trials = 1000

    # the CartPole-v0 is considered solved if 
    # for 100 consecutive trials, he cart pole has not 
    # fallen over and it has achieved an average 
    # reward of 195.0 
    # a reward of +1 is provided for every timestep 
    # the pole remains upright
    win_reward = { 'CartPole-v0' : 195.0 }

    env_id = 'CartPole-v0'
    # stores the reward per episode
    scores = deque(maxlen=win_trials)

    logger.setLevel(logger.ERROR)
    env = gym.make(env_id)

    outdir = "/tmp/dqn-%s" % env_id

    env = wrappers.Monitor(env,
                               directory=outdir,
                               video_callable=False,
                               force=True)

    env.seed(0)

    # instantiate the DQN/DDQN agent
    agent = DQNAgent(env.observation_space, env.action_space, epsilon = 0.01)
    agent.load_weights()

    # should be solved in this number of episodes
    episode_count = 10
    state_size = env.observation_space.shape[0]
    batch_size = 64

    # by default, CartPole-v0 has max episode steps = 200
    # you can use this to experiment beyond 200
    # env._max_episode_steps = 4000

    # Q-Learning sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        times = 0
        while not done:
            env.render()

            # in CartPole-v0, action=0 is left and action=1 is right
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # in CartPole-v0:
            # state = [pos, vel, theta, angular speed]
            next_state = np.reshape(next_state, [1, state_size])
            # store every experience unit in replay buffer
            # agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            times += 1

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                .format(episode, episode_count-1, times, agent.epsilon))
                print("episode_reward:", total_reward)


        # call experience relay
        # if len(agent.memory) >= batch_size:
        #     agent.replay(batch_size)
    
        scores.append(total_reward)

        # plot rewards 
        plot(scores, 'test_reward.png')
        mean_score = np.mean(scores)
        # if mean_score >= win_reward[args.env_id] \
        #         and episode >= win_trials:
        #     print("Solved in episode %d: \
        #            Mean survival = %0.2lf in %d episodes"
        #           % (episode, mean_score, win_trials))
        #     print("Epsilon: ", agent.epsilon)
        #     agent.save_weights()
        #     break
        # if (episode + 1) % win_trials == 0:
        #     print("Episode %d: Mean survival = \
        #            %0.2lf in %d episodes" %
        #           ((episode + 1), mean_score, win_trials))
            
    # agent.save_weights()

    # close the env and write monitor result info to disk
    env.close() 
    exit(0)

def train():

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id',
                        nargs='?',
                        default='CartPole-v0',
                        help='Select the environment to run')
    parser.add_argument("-d",
                        "--ddqn",
                        action='store_true',
                        help="Use Double DQN")
    parser.add_argument("-r",
                        "--no-render",
                        action='store_true',
                        help="Disable rendering (for env w/o graphics")
    args = parser.parse_args()

    # the number of trials without falling over
    win_trials = 1000

    # the CartPole-v0 is considered solved if 
    # for 100 consecutive trials, he cart pole has not 
    # fallen over and it has achieved an average 
    # reward of 195.0 
    # a reward of +1 is provided for every timestep 
    # the pole remains upright
    win_reward = { 'CartPole-v0' : 195.0 }

    # stores the reward per episode
    scores = deque(maxlen=win_trials)

    logger.setLevel(logger.ERROR)
    env = gym.make(args.env_id)

    outdir = "/tmp/dqn-%s" % args.env_id
    if args.ddqn:
        outdir = "/tmp/ddqn-%s" % args.env_id

    if args.no_render:
        env = wrappers.Monitor(env,
                               directory=outdir,
                               video_callable=False,
                               force=True)
    else:
        env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    # instantiate the DQN/DDQN agent
    if args.ddqn:
        agent = DDQNAgent(env.observation_space, env.action_space)
    else:
        agent = DQNAgent(env.observation_space, env.action_space)
        print("dqn")
        # exit(0)

    
    # agent.load_weights()
    # should be solved in this number of episodes
    episode_count = 500
    state_size = env.observation_space.shape[0]
    batch_size = 64

    # by default, CartPole-v0 has max episode steps = 200
    # you can use this to experiment beyond 200
    # env._max_episode_steps = 4000

    # Q-Learning sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        times = 0
        while not done:
            env.render()

            # in CartPole-v0, action=0 is left and action=1 is right
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)


            
            # in CartPole-v0:
            # state = [pos, vel, theta, angular speed]
            next_state = np.reshape(next_state, [1, state_size])
            # store every experience unit in replay buffer
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            times += 1

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                .format(episode, episode_count-1, times, agent.epsilon))
                print("episode_reward:", total_reward)


        # call experience relay
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)
    
        scores.append(total_reward)

        # plot rewards 
        plot(scores)
        mean_score = np.mean(scores)
        if mean_score >= win_reward[args.env_id] \
                and episode >= win_trials:
            print("Solved in episode %d: \
                   Mean survival = %0.2lf in %d episodes"
                  % (episode, mean_score, win_trials))
            print("Epsilon: ", agent.epsilon)
            agent.save_weights()
            break
        if (episode + 1) % win_trials == 0:
            print("Episode %d: Mean survival = \
                   %0.2lf in %d episodes" %
                  ((episode + 1), mean_score, win_trials))
            
    agent.save_weights()

    # close the env and write monitor result info to disk
    env.close() 


    return 0

if __name__ == '__main__':
    # test_model()


    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id',
                        nargs='?',
                        default='CartPole-v0',
                        help='Select the environment to run')
    parser.add_argument("-d",
                        "--ddqn",
                        action='store_true',
                        help="Use Double DQN")
    parser.add_argument("-r",
                        "--no-render",
                        action='store_true',
                        help="Disable rendering (for env w/o graphics")
    args = parser.parse_args()

    # the number of trials without falling over
    win_trials = 1000

    # the CartPole-v0 is considered solved if 
    # for 100 consecutive trials, he cart pole has not 
    # fallen over and it has achieved an average 
    # reward of 195.0 
    # a reward of +1 is provided for every timestep 
    # the pole remains upright
    win_reward = { 'CartPole-v0' : 195.0 }

    # stores the reward per episode
    scores = deque(maxlen=win_trials)

    logger.setLevel(logger.ERROR)
    env = gym.make(args.env_id)

    outdir = "/tmp/dqn-%s" % args.env_id
    if args.ddqn:
        outdir = "/tmp/ddqn-%s" % args.env_id

    if args.no_render:
        env = wrappers.Monitor(env,
                               directory=outdir,
                               video_callable=False,
                               force=True)
    else:
        env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    # instantiate the DQN/DDQN agent
    if args.ddqn:
        agent = DDQNAgent(env.observation_space, env.action_space)
    else:
        agent = DQNAgent(env.observation_space, env.action_space)
        print("dqn")
        # exit(0)

    
    # agent.load_weights()
    # should be solved in this number of episodes
    episode_count = 500
    state_size = env.observation_space.shape[0]
    batch_size = 64

    # by default, CartPole-v0 has max episode steps = 200
    # you can use this to experiment beyond 200
    # env._max_episode_steps = 4000

    # Q-Learning sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        times = 0
        while not done:
            env.render()

            # in CartPole-v0, action=0 is left and action=1 is right
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # in CartPole-v0:
            # state = [pos, vel, theta, angular speed]
            next_state = np.reshape(next_state, [1, state_size])
            # store every experience unit in replay buffer
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            times += 1

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                .format(episode, episode_count-1, times, agent.epsilon))
                print("episode_reward:", total_reward)


        # call experience relay
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)
    
        scores.append(total_reward)

        # plot rewards 
        plot(scores)
        mean_score = np.mean(scores)
        if mean_score >= win_reward[args.env_id] \
                and episode >= win_trials:
            print("Solved in episode %d: \
                   Mean survival = %0.2lf in %d episodes"
                  % (episode, mean_score, win_trials))
            print("Epsilon: ", agent.epsilon)
            agent.save_weights()
            break
        if (episode + 1) % win_trials == 0:
            print("Episode %d: Mean survival = \
                   %0.2lf in %d episodes" %
                  ((episode + 1), mean_score, win_trials))
            
    agent.save_weights()

    # close the env and write monitor result info to disk
    env.close() 
