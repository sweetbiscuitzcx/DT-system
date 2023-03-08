# coding=gbk
import time

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import os
import subprocess
import threading
import shutil

import envroom
from agents import *
from memories import PrioritizedNStepMemory
from networks import NoisyDuelingNetwork
# action=[14.5+x*0.2 for x in range(0,64) ]
# action1=np.arange(14.5,27.1,0.2)
# print(len(action))
env=envroom.prem()
mem = PrioritizedNStepMemory(int(10e5), n=3, update_every=30, gamma=0.99)
a = DQNAgent(state_size=env.statesize, hidden_sizes=[256, 256],
             action_size=env.actionsize, replay_memory=mem,
             double=True, Architecture=NoisyDuelingNetwork)



# print(type(action))
# print(type(action1))
# print(action1)
# print(action)
# print(a)

Reward_history=[]
eposideFlag = True


all_log_path = ['DT/out/logs/', 'DT/out/csvfile/', 'DT/out/tensorboard', 'DT/out/savemodel/local/',
                    'DT/out/savemodel/target/']
for pathName in all_log_path:
    if not os.path.exists(pathName):
        os.makedirs(pathName)

while True:

    if os.path.exists(all_log_path[1]+'RHfile.csv'):
        os.remove(all_log_path[1]+'RHfile.csv')

    # Initialization of rewards and fan_power in a round
    rewards_history = []
    # Select initial action
    startAction = [30.1 ]
    # Simulation environment initialization
    env.make(startAction)

    # time.sleep(0.1)
    stepFlag = True

    while True:
        # Get the current environment
        state = env.state()  # 12步
        # Get the current action according to the environment
        action = a.act(state)
        # Modify the action and execute the simulation environment
        state_next, reward, done ,data_step_save= env.steprun(action, a.t_step)
        # # Recalculate if model floating-point overflow
        state_flag = np.mean(state_next)
        if state_flag > 100 or state_flag < 1:
            stepFlag = False
            break
        # Store the reward and fan_power of each step in a round
        rewards_history.append(reward)  # reward_spectral

        print("timestep", a.t_step, "action:", action, "reward:", reward
              )
        # Store data and train models
        a.step(state, action, reward, state_next, done)  # reward_spectral

        # Judge whether to end the round
        if done is True:
            break


        # If the model diverges, this round will not be recorded
    if stepFlag is False:
        eposideFlag = False

    if eposideFlag is False:
        eposideFlag = True

    episode = a.episodes - 1

    Reward_history.append(sum(rewards_history))

    print('episode', episode, 'score', sum(rewards_history), 'score_max', max(Reward_history)
          )
    env.stepTotal=0
    # save episode_reward_history and each_episode_rewards_history

    name = [str(episode)]
    savefilepath='DT/out/'
    pd.DataFrame(data=[rewards_history], index=name). \
        to_csv(savefilepath+'each_episode_rewards_history.csv', sep=',', mode='a', encoding='utf-8')

    pd.DataFrame(data=[Reward_history], index=name). \
        to_csv(savefilepath+'episode_reward_history.csv', sep=',', encoding='utf-8')

    pd.DataFrame(data=np.array([a.losses]).reshape(-1, 1)). \
        to_csv(savefilepath+'Loss_history.csv', encoding='utf-8')
    np.savetxt('DT/out/csvfile/data_step_save_'+str(episode) +'.csv', data_step_save, delimiter=',', fmt='%.4f')
    # Save RHfile.csv history of each round as RHfile_episode.csv
    dstFile = all_log_path[1] + 'RHfile_' + str(episode) + '.csv'



    # Save model super parameters
    if episode % 100 == 0:
        # a.network_local.save_weights(all_log_path[3] + 'RHfile_' + str(episode))
        a.network_target.save_weights(all_log_path[4] + 'RHfile_' + str(episode))

    # Judge the conditions for model convergence and training completion
    if len(Reward_history) >= 10 and (np.mean(Reward_history[-10:])) > 20:
        print("Solved at episode {}!".format(episode))
        break