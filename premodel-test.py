# coding=gbk

import tensorflow as tf
import pandas as pd
import numpy as np
from tools import create_dataset
from metrics import All_Metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from normalization import MinMax01Scaler

new_model=tf.keras.models.load_model("savemodel/my_model12-1.h5")
new_model.summary()



data_all_save = pd.read_csv('./start.csv', header=None)
data_all = np.array(data_all_save)
# 归一化
startAction = [37.5]
data_all[11][1]=startAction[0]

minimum = -11
maximum = 143
scaler = MinMax01Scaler(minimum, maximum)
data_all = scaler.transform(data_all)
print(data_all.shape)

data_test = data_all
print(data_test.shape)
time_step=12
pre_step=1
# test
batch_size =1
test_X, Y_truth = create_dataset(data_test, time_step, pre_step)
acc_num=[]
startAction = [37.5]
for i in range(test_X.shape[0] // batch_size):

    test_X_test = []
    Y_truth_test = []
    X = test_X[i:i + batch_size, :, :]
    Y = Y_truth[i:i + batch_size, :]
    # print(X.shape)
    y_hat = new_model.predict(X)
    # print(y_hat.shape)

    y_hat = y_hat.reshape(pre_step, -1)#此处做修改，原来为batch——size，现为所预测步数
    # print(y_hat.shape)
    Y_truth_test.append(Y[:, :])
    Y_truth_test = np.array(Y_truth_test)
    Y_truth_test = Y_truth_test.reshape(y_hat.shape[0], -1)

    Y_truth_test = scaler.inverse_transform(Y_truth_test)
    y_hat = scaler.inverse_transform(y_hat)
    # y_hat_reshape=y_hat.reshape(-1,8)

    Y_truth_test1=sum(Y_truth_test[:,-4:])
    y_hat1=sum(y_hat[:,-4:])
    # aa=sum(Y_truth_test1)
    # bb=sum(y_hat1)
    aa=Y_truth_test1
    bb=y_hat1
    acc1=abs(bb[0]-aa[0])/aa[0]*100
    acc2 = abs(bb[1] - aa[1]) / aa[1] * 100
    acc3 = abs(bb[2] - aa[2]) / aa[2] * 100
    acc4 = abs(bb[3] - aa[3]) / aa[3] * 100
    acc= (acc1+acc2+acc3+acc4)/4
    acc_num.append(acc)

    if i//11==0 and i<240:
        np.savetxt(f'DT/pre_{i}.csv', y_hat, delimiter=',', fmt='%.4f')
        np.savetxt(f'DT/true_{i}.csv', Y_truth_test, delimiter=',', fmt='%.4f')
np.savetxt(f'each-accnumber.csv',acc_num,delimiter=',',fmt='%.4f')

# data_all_save = pd.read_csv('./start.csv', header=None)
# a=data_all_save.iloc[-1,:]
# print(type(a))