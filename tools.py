# coding=gbk
import tensorflow as tf
import pandas as pd
import numpy as np
from metrics import All_Metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from normalization import MinMax01Scaler

def discounted_returns(returns, gamma=0.5):
    G = 0
    for i in range(len(returns)):
        G += returns[i] * gamma ** i
    return G

def create_dataset(data, n_predictions, n_next):
    '''
    对数据进行处理,
    用于对有真实值的测试对比
    '''
    dim = data.shape[1]
    train_X, train_Y = [], []
    for i in range(data.shape[0] - n_predictions - n_next +1):
        a = data[i:(i + n_predictions), :]
        train_X.append(a)
        tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
        b = []
        '''
        for j in range(len(tempb)):
            for k in [0,2,3,8,11,12,13,14,18,19,20,21,23,24,25,27]:
                b.append(tempb[j,k])
        '''
        for j in range(len(tempb)):
            for k in range(tempb.shape[1]):
                b.append(tempb[j, k])

        train_Y.append(b)
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')

    return train_X, train_Y

def create_dataset_env(data, n_predictions, n_next):
    '''
    对数据进行处理
    '''

    train_X, train_Y = [], []
    for i in range(data.shape[0] - n_predictions - n_next +2):
        a = data[i:(i + n_predictions), :]
        train_X.append(a)
        # tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
        # b = []
        # for j in range(len(tempb)):
        #     for k in range(tempb.shape[1]):
        #         b.append(tempb[j, k])

        # train_Y.append(b)
    train_X = np.array(train_X, dtype='float64')
    # train_Y = np.array(train_Y, dtype='float64')

    return train_X

def steprun_dataset_env(data, n_predictions, n_next, ):
    '''
    对数据进行处理
    '''
    dim = data.shape[1]
    use_X= []
    i =data.shape[0] - n_predictions - n_next +1
    a = data[i:(i + n_predictions), :]
    use_X.append(a)
    use_X = np.array(use_X, dtype='float64')
    return use_X