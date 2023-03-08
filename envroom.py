# coding=gbk

import tensorflow as tf
import pandas as pd
import numpy as np
from tools import create_dataset,create_dataset_env,steprun_dataset_env
from metrics import All_Metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from normalization import MinMax01Scaler
from agents import *
#读取储存好的预测模型
new_model=tf.keras.models.load_model("savemodel/my_model12-1.h5")
# 读取用于模型初始化的预测文件
data_all = pd.read_csv('./start.csv', header=None)
data_all = np.array(data_all)
# 归一化模块
# minimum = -11
# maximum = 143
# scaler = MinMax01Scaler(minimum, maximum)
# data_all = scaler.transform(data_all)
# print(data_all.shape)
data_test = data_all
# print(data_test.shape)
time_step=12
pre_step=1
batch_size =1

class prem(object):
    def __init__(self, statesize=4, physicalTimeStep=1):
        self.minimum = -11
        self.maximum = 143
        self.time_step = 12
        self.pre_step = 1
        self.batch_size = 1
        self.statesize = statesize  # Number of humidity sensors
        self.physicalTimeStep = physicalTimeStep
        self.stepTotal=1
        self.j=0
        self.scaler = MinMax01Scaler(self.minimum, self.maximum)
        self.actionsize=160
    def make (self,action):
        #归一化
        data_all = pd.read_csv('./start.csv', header=None)
        data_all = np.array(data_all)
        data_all[time_step-1][1]=action[0]
        data_all = self.scaler.transform(data_all)
        data_test = data_all
        test_X = create_dataset_env(data_test, time_step, pre_step)

        test_X_test = []
        X = test_X[0:0 + batch_size, :, :]
        # print(X.shape)
        y_hat = new_model.predict(X)
        # print(y_hat.shape)
        y_hat = y_hat.reshape(pre_step, -1)  # 此处做修改，原来为batch——size，现为所预测步数
        # print(y_hat.shape)
        y_hat = self.scaler.inverse_transform(y_hat)
        data_all = self.scaler.inverse_transform(data_test)
        data_start_save = np.r_[data_all, y_hat]
        np.savetxt(f'DT/data_start_save.csv', data_start_save, delimiter=',', fmt='%.4f')
        np.savetxt(f'DT/now_state.csv', y_hat, delimiter=',', fmt='%.4f')

    def state(self):
        now_state=pd.read_csv('DT/now_state.csv', header=None)
        now_state=np.array(now_state)
        now_state=now_state[:,-4:]
        # state = [float('{:.2f}'.format(p)) for p in now_state]
        # state = np.round(now_state, decimals=2)#保留两位小数
        state = now_state.tolist()
        return state
    def steprun(self,action, step):
        if step ==0:
            data_step=pd.read_csv('DT/data_start_save.csv', header=None)
        else:
            data_step=pd.read_csv('DT/data_step_save.csv', header=None)
        data_step1 = np.array(data_step)
        data_step1[-1][1]=action[0]
        data_step1 = self.scaler.transform(data_step1)
        data_test_step = data_step1
        use_X = steprun_dataset_env(data_test_step, time_step, pre_step)
        X = use_X[0:0 + batch_size, :, :]
        y_hat = new_model.predict(X)
        y_hat = y_hat.reshape(pre_step, -1)  # 此处做修改，原来为batch——size，现为所预测步数
        y_hat = self.scaler.inverse_transform(y_hat)
        data_all=self.scaler.inverse_transform(data_step1)
        data_step_save=np.r_[data_all,y_hat]
        np.savetxt(f'DT/data_step_save.csv', data_step_save, delimiter=',', fmt='%.4f')
        np.savetxt(f'DT/now_state.csv', y_hat, delimiter=',', fmt='%.4f')
        step=step+1
        next_s = y_hat[:, -4:]
        # next_s = [float('{:.2f}'.format(p)) for p in next_s]
        next_s = next_s.tolist()
        reward1 = 0
        reward2 = 0
        reward3 = 0
        reward4 = 0
        if next_s[0][0] < 19 or next_s[0][0] > 22:
            reward1= -10
        elif next_s[0][0]>=19 and next_s[0][0]<19.5:
            reward1=2
        elif next_s[0][0]>=19.5 and next_s[0][0]<20.5:
            reward1=4
        elif next_s[0][0]>=20.5 and next_s[0][0]<21.5:
            reward1=2
        elif next_s[0][0]>=21.5 and next_s[0][0]<=22:
            reward1=1

        if next_s[0][1] < 19 or next_s[0][1] > 22:
            reward2 = -10
        elif next_s[0][1]>= 19  and next_s[0][1] < 19.5:
            reward2 = 2
        elif next_s[0][1] >= 19.5 and next_s[0][1] < 20.5:
            reward2 = 4
        elif next_s[0][1] >= 20.5 and next_s[0][1] < 21.5:
            reward2 = 2
        elif next_s[0][1] >= 21.5 and next_s[0][1] <= 22:
            reward2 = 1

        if next_s[0][2] < 19 or next_s[0][2] > 22:
            reward3 = -10
        elif next_s[0][2] >= 19 and next_s[0][2] < 19.5:
            reward3 = 2
        elif next_s[0][2] >= 19.5 and next_s[0][2] < 20.5:
            reward3 = 4
        elif next_s[0][2] >= 20.5 and next_s[0][2] < 21.5:
            reward3 = 2
        elif next_s[0][2] >= 21.5 and next_s[0][2] <= 22:
            reward3 = 1

        if next_s[0][3] < 19 or next_s[0][3] > 22:
            reward4 = -10
        elif next_s[0][3] >= 19 and next_s[0][3] < 19.5:
            reward4 = 2
        elif next_s[0][3] >= 19.5 and next_s[0][3] < 20.5:
            reward4 = 4
        elif next_s[0][3] >= 20.5 and next_s[0][3] < 21.5:
            reward4 = 2
        elif next_s[0][3] >= 21.5 and next_s[0][3] <= 22:
            reward4 = 1

        reward=reward1+reward2+reward3+reward4


        if self.stepTotal <= self.physicalTimeStep * 50:
            done = False
            self.stepTotal=self.stepTotal+1
        else :
            done = True
        return next_s, reward, done,data_step_save



