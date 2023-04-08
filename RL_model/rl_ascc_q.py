import numpy as np
import random
from collections import defaultdict
import tools_ascc


class QLearningAgent:
    def __init__(self, actions, epsilon = 1.0, episodes=500, memory_size = 512):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.01
        self.learning_rate_min = 0.0001
        self.learning_rate_decay = 0.000001 # 1e-8
        
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])

        # discount rate
        self.gamma = 0.9

        # initially 90% exploration, 10% exploitation
        self.epsilon = epsilon
        # iteratively applying decay til 
        # 10% exploration/90% exploitation
        self.epsilon_min = 0.01
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** \
                             (1. / float(episodes))
        
        self.memory = []
        self.memory_size = memory_size

        self.replay_counter = 0

    def load_weights(self, q_table_path = "ascc_q_table.txt"):
        """save Q Network params to a file"""
        # self.q_model.load_weights(self.weights_file)

        new_qtable = tools_ascc.load_dict(q_table_path)

        for k, v in new_qtable.items():
            self.q_table[k] = v

        print('load q_table:', len(self.q_table))

    # 采样 <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # 贝尔曼方程更新
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

    # 从Q-table中选取动作
    def act(self, state):
        if np.random.rand() < self.epsilon:
            # 贪婪策略随机探索动作
            action = np.random.choice(self.actions)
        else:
            # 从q表中选择
            # if state not in self.q_table.keys():
            #     action = 1
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)
    
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

            self.learn(state, action, reward, next_state)


        # update exploration-exploitation probability
        self.update_epsilon()

        self.replay_counter += 1

    def update_epsilon(self):
        """decrease the exploration, increase exploitation"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.learning_rate > self.learning_rate_min:
            self.learning_rate -= self.learning_rate_decay

# if __name__ == "__main__":
#     env = Env()
#     agent = QLearningAgent(actions=list(range(env.n_actions)))
#     for episode in range(1000):
#         state = env.reset()
#         while True:
#             env.render()
#             # agent产生动作
#             action = agent.get_action(str(state))
#             next_state, reward, done = env.step(action)
#             # 更新Q表
#             agent.learn(str(state), action, reward, str(next_state))
#             state = next_state
#             env.print_value_all(agent.q_table)
#             # 当到达终点就终止游戏开始新一轮训练
#             if done:
#                 break
