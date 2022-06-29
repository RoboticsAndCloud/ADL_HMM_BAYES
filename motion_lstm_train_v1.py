import argparse
from operator import mod
from os import stat
import os
from statistics import mode

import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
from prettytable import PrettyTable

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation

import motion_env_ascc

from datetime import datetime
from datetime import timedelta

from motion_env_ascc import DEBUG
from timeit import default_timer as timer
import tools_ascc
import argparse
import time


MILAN_DATASET_DAYS = 61 # the days have activity data, totally 82 includes 61 + 21 no data
DATASET_DAYS = MILAN_DATASET_DAYS
DATASET_TRAIN_DAYS = 82 # includes the days without acitivity data, totally 82 days

DATASET_TEST_DAYS = DATASET_DAYS - DATASET_TRAIN_DAYS

MILAN_BASE_DATE = '2009-10-16'

MILAN_BASE_DATE_HOUR = '2009-10-16 06:00:00'


DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
HOUR_TIME_FORMAT = "%H:%M:%S"
DAY_FORMAT_STR = '%Y-%m-%d'

model_path = './model/model_motion_lstm'

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


g_i = 0

GAMMA = 0.99
HIDDEN_SIZE = 128
# env = gym.make('CartPole-v1')
# env.seed(1)
torch.manual_seed(1)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()

# MILAN_BASE_DATE
env = motion_env_ascc.EnvASCC(MILAN_BASE_DATE_HOUR)

ACTIONS_SIZE = 17


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.lstm = nn.LSTMCell(2, HIDDEN_SIZE)
        # self.affine = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.action_head = nn.Linear(HIDDEN_SIZE, ACTIONS_SIZE)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)
        self.saved_actions = []
        self.rewards = []
        self.reset()
        
    def reset(self):
        self.hidden = Variable(torch.zeros(1, HIDDEN_SIZE)), Variable(torch.zeros(1, HIDDEN_SIZE))

    def detach_weights(self):
        self.hidden = self.hidden[0].detach(), self.hidden[1].detach()

    def forward(self, x):
        x = x.unsqueeze(0)
        self.hidden = self.lstm(x, self.hidden)
        x = self.hidden[0]
        x = x.squeeze(0)
        # x = self.affine(x)
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


model = Policy()

count_parameters(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)
eps = np.finfo(np.float32).eps.item()

episilon_from_prior = 0.9

# 54 actions
def select_action_prior_knowledge(state, prior_action):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    # print('select_action: probs:',probs)

    m = Categorical(probs)
    action = m.sample()
    print("action info:", action.item())

    try_max_cnt = 1000 * 1000 * 1000
    if np.random.rand() < episilon_from_prior:
        # action = prior_action # From prior knowledge
        i = 0
        while(action.item() != prior_action):
            # print("in while prior action:", prior_action, " action info:", action)
            if i > try_max_cnt:
                print('i > try_max_cnt')
                break

            i = i + 1
            
            action = m.sample()

    
    print("action info:", action.item())

    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    # print('select_action: probs:',probs)

    m = Categorical(probs)
    action = m.sample()
    if np.random.rand() < episilon_from_prior:
        action = 2 # From prior knowledge
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    #loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss = torch.stack(policy_losses).sum() 
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


"""
state: [-0.02626715  0.38390708  0.16591503 -0.02105215]
reward: 1.0
done: False
x: tensor([[-0.0263,  0.3839,  0.1659, -0.0211]])
probs: tensor([0.4828, 0.5172], grad_fn=<SoftmaxBackward>)
state_value: tensor([-0.0444], grad_fn=<AddBackward0>)
state: [-0.01858901  0.18684247  0.16549398  0.31904107]
reward: 1.0
done: False
x: tensor([[-0.0186,  0.1868,  0.1655,  0.3190]])
probs: tensor([0.4837, 0.5163], grad_fn=<SoftmaxBackward>)
state_value: tensor([-0.0429], grad_fn=<AddBackward0>)
state: [-0.01485216 -0.01020216  0.1718748   0.65900314]
reward: 1.0
done: False
x: tensor([[-0.0149, -0.0102,  0.1719,  0.6590]])
probs: tensor([0.4831, 0.5169], grad_fn=<SoftmaxBackward>)
state_value: tensor([-0.0370], grad_fn=<AddBackward0>)
state: [-0.01505621  0.18216385  0.18505487  0.42498842]
reward: 1.0
done: False
x: tensor([[-0.0151,  0.1822,  0.1851,  0.4250]])
probs: tensor([0.4829, 0.5171], grad_fn=<SoftmaxBackward>)
state_value: tensor([-0.0375], grad_fn=<AddBackward0>)
"""

def train_by_day(date_day_hour_time):

    env = motion_env_ascc.EnvASCC(date_day_hour_time)
    max_steps = 2000  # for 60 seconds interval
    max_steps = 1200  # for 120 seconds interval

    counter = 0
    done_counter = 0
    done_reward = 0

    running_reward = 10
    # for i_episode in count(1):
    state = env.reset()
    state = np.array(state, dtype=float)
    # print('state:', state)


    episode_reward = []

    global g_i
    g_i = g_i + 1
    for t in range(max_steps):
        # action = select_action(state)
        prio_action, time_cost = env.get_motion_triggerred_action()
        action = select_action_prior_knowledge(state, prio_action)


        state, reward, done = env.step(action)

        state = np.array(state, dtype=float)
        # print('state:', state)
        model.rewards.append(reward)
        episode_reward.append(reward)

        if done:
            model.reset()
            break
        else:
            model.detach_weights()

    running_reward = running_reward * 0.99 + t * 0.01

    finish_episode()

    print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            g_i, t, running_reward))


    last_action = env.action_space.sample() # todo Box.sample?
    if motion_env_ascc.DEBUG:
        print("Sampled last_action", last_action)
    max_steps = 2000  # for 60 seconds interval
    max_steps = 1200  # for 120 seconds interval

    counter = 0
    done_counter = t
    done_reward = 0
   

    # if done, do not break due to the problem: the lists like episode_action should be with the same size
    if done and done_counter == 0:
        done_counter = counter 

    if done and done_reward == 0:
        done_reward = np.sum(episode_reward)



    print("===================================================")
    print("Train counter:", counter)
    print("Train done_counter:", done_counter)
    print('Train day_by_day: Reward: ', np.sum(episode_reward))
    print('Train day_by_day done_reward: Reward: ', done_reward)

    
    if env.done:
        print("Activity_none_times:", env.activity_none_times)
        print("Expected_activity_none_times:", env.expected_activity_none_times)
        print("Hit times:", env.done_hit_event_times)
        print("Miss times:", env.done_missed_event_times)
        print("Random Miss times:", env.done_random_miss_event_times)
        print("Middle times:", env.done_middle_event_times)
        print("Penalty times:", env.done_penalty_times)
        print("Uncertain times:", env.done_uncertain_times)
        print("Total times:", env.done_totol_check_times)
        print("Residual power:", env.done_residual_power)
        print("Beginning event times:", env.done_beginning_event_times)
        print("Endding event times:", env.done_end_event_times)
        print("Middle event times:", env.done_middle_event_times)
        print("Day End Running time:", env.done_running_time)
        print("Reward:", env.done_reward)
        print("Done status:", env.done)
        print("Sensors Energy cost:", env.done_energy_cost)
        print("Sensors Time cost:", env.done_time_cost)
        print("Sensors Energy total  cost:", env.energy_cost)
        print("Sensors Time total cost:", env.time_cost)
        print("Total Time cost:", env.done_total_time_cost)
        print("Motion_triggered_times:", env.motion_triggered_times)
        end_time_of_wmu = datetime.strptime(env.done_running_time.strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
        print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)
    else:
        print("Activity_none_times:", env.activity_none_times)
        print("Expected_activity_none_times:", env.expected_activity_none_times)
        print("Hit times:", env.hit_event_times)
        print("Miss times:", env.missed_event_times)
        print("Random Miss times:", env.random_miss_event_times)
        print("Middle times:", env.done_middle_event_times)
        print("Penalty times:", env.done_penalty_times)    
        print("Uncertain times:", env.uncertain_times)
        print("Total times:", env.totol_check_times)
        print("Beginning event times:", env.beginning_event_times)
        print("Endding event times:", env.end_event_times)
        print("Middle event times:", env.middle_event_times)
        print("Residual power:", env.residual_power)
        print("Day End Running time:", env.running_time)
        print("Reward:", env.done_reward)
        print("Done status:", env.done)
        print("Sensors Energy cost:", env.energy_cost)
        print("Sensors Time cost:", env.time_cost)
        print("Sensors Energy total  cost:", env.energy_cost)
        print("Sensors Time total cost:", env.time_cost)
        print("Total Time cost:", env.done_total_time_cost)
        print("Motion_triggered_times:", env.motion_triggered_times)
        end_time_of_wmu = datetime.strptime(env.running_time.strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
        print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)


    
    end_time_of_wmu = datetime.strptime(env.done_running_time.strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
    print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)
    
    print("Activity_none_times \t Expected_activity_none_times \t Hit times \t Miss times \
        \t Random Miss times \t Penalty times \t Uncertain times \t Total times \
        \t Residual power \t Beginning event times \t Endding event times \t Middle event times \
        \t Day End Running time \t Done status \t Duration of Day \t DoneReward \t Reward  \
        \t Sensors energy cost \t Sensors time cost \t total time cost \t Motion_triggered_times \t")

    print(env.activity_none_times, '\t', env.expected_activity_none_times, '\t',env.done_hit_event_times, '\t', env.done_missed_event_times, \
        '\t', env.done_random_miss_event_times, '\t', env.done_penalty_times, '\t', env.done_uncertain_times, '\t', env.done_totol_check_times, \
        '\t', env.done_residual_power, '\t', env.done_beginning_event_times, '\t', env.done_end_event_times, '\t', env.done_middle_event_times, \
        '\t', env.done_running_time, '\t', env.done, '\t', (end_time_of_wmu - env.day_begin).seconds/3600.0, '\t', done_reward, '\t', np.sum(episode_reward), \
        '\t', env.done_energy_cost, '\t', env.done_time_cost, "\t", env.done_total_time_cost, "\t", env.motion_triggered_times)


        # print("Activity_none_times:", env.activity_none_times)
        # print("Expected_activity_none_times:", env.expected_activity_none_times)
        # print("Hit times:", env.done_hit_event_times)
        # print("Miss times:", env.done_missed_event_times)
        # print("Random Miss times:", env.done_random_miss_event_times)
        # # print("Middle times:", env.done_middle_event_times)
        # print("Penalty times:", env.done_penalty_times)
        # print("Uncertain times:", env.done_uncertain_times)
        # print("Total times:", env.done_totol_check_times)
        # print("Residual power:", env.done_residual_power)
        # print("Beginning event times:", env.done_beginning_event_times)
        # print("Endding event times:", env.done_end_event_times)
        # print("Middle event times:", env.done_middle_event_times)
        # print("Day End Running time:", env.done_running_time)
        # print("Done status:", env.done)
        # end_time_of_wmu = datetime.strptime(env.done_running_time.strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
        # print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)
        # print("Reward:", env.done_reward)

    print("Display information:")
    env.display_action_counter()
    env.display_info()
    print("===================================================")
    # sac_trainer.display_info()

    return np.sum(episode_reward), env.done



def save_model(path):
    torch.save(model.state_dict(), path+'_policy')


def load_model(path):
    model.load_state_dict(torch.load(path+'_policy'))
    model.eval()



def test(date_day_time):
    date_day_hour_time = date_day_time + " 00:00:00"

    env = motion_env_ascc.EnvASCC(date_day_hour_time)

    state = env.reset()  # next_state_ = [self.day_begin, self.initial_state, self.residual_power]


    last_action = env.action_space.sample() # todo Box.sample?
    if motion_env_ascc.DEBUG:
        print("Sampled last_action", last_action)

    episode_state = []
    episode_action = []
    episode_last_action = []
    episode_reward = []
    episode_next_state = []
    episode_done = []
    hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                  torch.zeros([1, 1, hidden_dim],
                              dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
    max_steps = 1200 * 5  # for test, we want to check how much times and how much energy it is needed to get the result
    # max_steps = 1200  # for 120 seconds interval

    counter = 0
    done_counter = 0
    done_reward = 0
    for step in range(max_steps):
        hidden_in = hidden_out

        if environment_ascc.DEBUG:
            print("==Policy state:", state, "last_action:", last_action)

        if (not env.done):
            start = timer()
            action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in,
                                   deterministic=DETERMINISTIC)
            print("==Policy new action:", action)
            print("==hidden_out:", hidden_out)

            end = timer()
            #print("Get_action time cost:", end-start)
        else:
            # if env.done, we use action 21 to finish the extral steps
            # Each day, we use default action 20 to train it first, it may be a good model, 
            # Just for acceleration
            print("Steps:", step)
            action = [0.2099999]
        
        if environment_ascc.DEBUG:
            print("==Policy new action:", action)

        next_state, reward, done = env.step(action)

        if environment_ascc.DEBUG:
            print("==Policy reward:", reward)

        # Example:
        # next_state: [0.1494 -0.909 -2.2445]
        # energy:  [Current Time,  Previous Activity， Residual Power]
        # next_state_ = [train_running_time, self.activity, self.residual_power]

        if step == 0:
            ini_hidden_in = hidden_in
            ini_hidden_out = hidden_out
        episode_state.append(state)
        episode_action.append(action)
        episode_last_action.append(last_action)
        episode_reward.append(reward)
        episode_next_state.append(next_state)
        episode_done.append(done)

        state = next_state
        last_action = action

        #print("batch_size len buffer:", len(replay_buffer))
        # if len(replay_buffer) > batch_size:
            # print("=====================Begin to Train")
            #if step % batch_size == 0:
            # for i in range(update_itr):
                # _ = sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY,
                                    #    target_entropy=-1. * action_dim)
        
        counter = counter + 1

        # if done, do not break due to the problem: the lists like episode_action should be with the same size
        if done and done_counter == 0:
            done_counter = counter 

        if done and done_reward == 0:
            done_reward = np.sum(episode_reward)

        if done:
            break


    # todo make the size of different batches list same
    # replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
                    #    episode_reward, episode_next_state, episode_done)

    print("===================================================")
    print("Test counter:", counter)
    print("Test done_counter:", done_counter)
    print('Test day_by_day: Reward: ', np.sum(episode_reward))
    print('Test day_by_day done_reward: Reward: ', done_reward)

    rewards.append(np.sum(episode_reward))
    
    if env.done:
        print("Activity_none_times:", env.activity_none_times)
        print("Expected_activity_none_times:", env.expected_activity_none_times)
        print("Hit times:", env.done_hit_event_times)
        print("Miss times:", env.done_missed_event_times)
        print("Random Miss times:", env.done_random_miss_event_times)
        print("Middle times:", env.done_middle_event_times)
        print("Penalty times:", env.done_penalty_times)
        print("Uncertain times:", env.done_uncertain_times)
        print("Total times:", env.done_totol_check_times)
        print("Residual power:", env.done_residual_power)
        print("Beginning event times:", env.done_beginning_event_times)
        print("Endding event times:", env.done_end_event_times)
        print("Middle event times:", env.done_middle_event_times)
        print("Day End Running time:", env.done_running_time)
        print("Reward:", env.done_reward)
        print("Done status:", env.done)
        print("Sensors Energy cost:", env.done_energy_cost)
        print("Sensors Time cost:", env.done_time_cost)
        print("Sensors Energy total  cost:", env.energy_cost)
        print("Sensors Time total cost:", env.time_cost)
        print("Total Time cost:", env.done_total_time_cost)
        print("Motion_triggered_times:", env.motion_triggered_times)
        #print("Done_middle_event_times_uncertain:", env.done_middle_event_times_uncertain)
        print("Hit_activity_check_times", env.hit_activity_check_times)
        end_time_of_wmu = datetime.strptime(env.done_running_time.strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
        print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)

    else:
        print("Activity_none_times:", env.activity_none_times)
        print("Expected_activity_none_times:", env.expected_activity_none_times)
        print("Hit times:", env.hit_event_times)
        print("Miss times:", env.missed_event_times)
        print("Random Miss times:", env.random_miss_event_times)
        print("Middle times:", env.done_middle_event_times)
        print("Penalty times:", env.done_penalty_times)    
        print("Uncertain times:", env.uncertain_times)
        print("Total times:", env.totol_check_times)
        print("Beginning event times:", env.beginning_event_times)
        print("Endding event times:", env.end_event_times)
        print("Middle event times:", env.middle_event_times)
        print("Residual power:", env.residual_power)
        print("Day End Running time:", env.running_time)
        print("Reward:", env.done_reward)
        print("Done status:", env.done)
        print("Sensors Energy cost:", env.energy_cost)
        print("Sensors Time cost:", env.time_cost)
        print("Sensors Energy total  cost:", env.energy_cost)
        print("Sensors Time total cost:", env.time_cost)
        print("Total Time cost:", env.done_total_time_cost)
        print("Motion_triggered_times:", env.motion_triggered_times)
        end_time_of_wmu = datetime.strptime(env.running_time.strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
        print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)


    
    end_time_of_wmu = datetime.strptime(env.done_running_time.strftime(DATE_TIME_FORMAT).split()[1], HOUR_TIME_FORMAT)
    print("Duration of Day:", (end_time_of_wmu - env.day_begin).seconds/3600.0)
    

    print("Display information:")
    env.display_action_counter()
    env.display_info()
    print("===================================================")
    sac_trainer.display_info()

    print("Activity_none_times \t Expected_activity_none_times \t Hit times \t Miss times \
        \t Random Miss times \t Penalty times \t Uncertain times \t Total times \
        \t Residual power \t Beginning event times \t Endding event times \t Middle event times \
        \t Day End Running time \t Done status \t Duration of Day \t DoneReward \t Reward  \
        \t Sensors energy cost \t Sensors time cost \t total time cost \t Motion_triggered_times \t Hit_activity_check_times \t motion_activity_cnt \t")

    print(env.activity_none_times, '\t', env.expected_activity_none_times, '\t',env.done_hit_event_times, '\t', env.done_missed_event_times, \
        '\t', env.done_random_miss_event_times, '\t', env.done_penalty_times, '\t', env.done_uncertain_times, '\t', env.done_totol_check_times, \
        '\t', env.done_residual_power, '\t', env.done_beginning_event_times, '\t', env.done_end_event_times, '\t', env.done_middle_event_times, \
        '\t', env.done_running_time, '\t', env.done, '\t', (end_time_of_wmu - env.day_begin).seconds/3600.0, '\t', done_reward, '\t', np.sum(episode_reward), \
        '\t', env.done_energy_cost, '\t', env.done_time_cost, "\t", env.done_total_time_cost, "\t", env.motion_triggered_times, '\t', env.hit_activity_check_times, '\t',env.motion_activity_cnt)

    print("===================================================")

    return episode_reward, env.done


def train(date_day_time):
    train_times_each_day = 10 * 2
    # train_times_each_day = 2
    date_day_hour_time = date_day_time + " 00:00:00"
    
    train_rewards = []
    res_l = []
    res_done_times = 0
    for i in range(train_times_each_day):
        start = timer()
        train_reward, done = train_by_day(date_day_hour_time)
        train_rewards.append(train_reward)
        end = timer()
        # print("Train_by_day times:", i, " Train time:", end-start)
        res_l.append(done)
        if done:
            res_done_times = res_done_times + 1
    
    # save_model(model_path)
    # exit(0)
    # if res_done_times >= 6:
    #     return train_rewards

    # res_done_times = 0

    # # make sure each day be trained correctly
    # while(res_done_times < 6):
    #     train_reward, done = train_by_day(date_day_hour_time)
    #     if done:
    #         res_done_times = res_done_times + 1

    end = timer()
    print("Train_by_day times:", i, " Train time:", end-start)

    return train_rewards

def plot(rewards, day_time):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    file = './plot_figure/motion_lstm_' + day_time + '.png'
    plt.savefig(file)
    # plt.show()

if __name__ == '__main__':
    if args.train:
        # training loop
        start = timer()
        print("Start:", start)
        for eps in range(1):
            # if eps == 1:
            #     exit(0)

            counter = 0
            episode_reward = []
            for i in range(DATASET_TRAIN_DAYS):
            # for i in range(DATASET_TRAIN_DAYS - 15):
            # for i in range(2):

                # if i == 1:
                #    exit(0)

                base_date = MILAN_BASE_DATE
                day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days=i)
                day_time_str = day_time_train.strftime(DAY_FORMAT_STR)

                activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
                    tools_ascc.get_activity_date(day_time_str)

                if len(activity_begin_list) == 0:
                    continue

                # train the model by one single day
                # date_day_hour_time
                train_rewards = train(day_time_str)
                print("day_time_str:", day_time_str)
                plot(train_rewards, day_time_str)
                episode_reward.extend(train_rewards)
                # exit(0)
                #train_reward = train_from_model(day_time_str)

                counter = counter + 1

                try:
                    print("Save the model:", day_time_str)
                    new_model_dir = './model_' + day_time_str
                    if not os.path.exists(new_model_dir):
                        os.mkdir(new_model_dir)
                    new_model_path = './model_' + day_time_str + '/model_motion_lstm'

                    print("Save the model success:", new_model_path)
                    save_model(new_model_path)

                    print("Save the model success:", new_model_path)
                except Exception as e:
                    print("Error Save the model:", day_time_str)
                    print(e.message)
                print("Save the model:", day_time_str)

                end = timer()
                print("DATASET_TRAIN_DAYS end:", end)
                print("Train time each day:", end - start)
            print("Total days:", counter)
            plot(episode_reward, "final")
            print("final reward:")
            print(episode_reward)

        save_model(model_path)

        end = timer()
        print("Train time:",end - start)

    if args.test:
        load_model(model_path)
        for eps in range(5):
            counter = 0
            for i in range(30):
                base_date = MILAN_BASE_DATE
                # 0 + 55: 1210
                day_time_train = datetime.strptime(base_date, DAY_FORMAT_STR) + timedelta(days = i + 56)
                day_time_str = day_time_train.strftime(DAY_FORMAT_STR)
                print(day_time_str)
                # exit(0)


                activity_date_dict, activity_begin_dict, activity_end_dict, activity_begin_list, activity_end_list = \
                    tools_ascc.get_activity_date(day_time_str)

                if len(activity_begin_list) == 0:
                    continue

                # train the model by one single day
                test(day_time_str)

                exit(0)

                counter = counter + 1

            print("Total days:", counter)

            print('Episode: ', eps)
