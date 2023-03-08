# coding=gbk

import tensorflow as tf
import pandas as pd
import numpy as np
from tools import create_dataset
from metrics import All_Metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from normalization import MinMax01Scaler


new_model=tf.keras.models.load_model("savemodel/my_model12-1-1-notime.h5")
new_model.compile()



new_model.summary()


data_all = pd.read_csv('data/MM-1-notime.csv', header=None)
data_all = np.array(data_all)
# 归一化
minimum = data_all.min()
maximum = data_all.max()
scaler = MinMax01Scaler(minimum, maximum)
data_all = scaler.transform(data_all)
print(data_all.shape)

# data_train = data_all[:int(len(data_all) * 0.8), :]


time_step=12
pre_step=1
# test
batch_size =1
from tensorflow.python.keras.models import load_model
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# 仅对最后200条数据进行测试 因为预测仅最新有作用
# data_test = data_all[int(len(data_all)*0.7):,:]
data_test = data_all[int(len(data_all) * 0.8):, :]

print(data_test.shape)

# print(y_hat.dtype)

test_X, Y_truth = create_dataset(data_test, time_step, pre_step)

# print(test_X.shape)
# print(Y_truth.shape)
# Y_pred = []
mae_list = []
rmse_list = []
mape_list = []
rrse_list = []
corr_list = []
test_X_test = []
acc_num=[]
line=0
line1=0
line2=0
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
    # test_X_test = np.swapaxes(test_X_test, 1, 0)
    Y_truth_test = np.array(Y_truth_test)
    Y_truth_test = Y_truth_test.reshape(y_hat.shape[0], -1)

    Y_truth_test = scaler.inverse_transform(Y_truth_test)
    y_hat = scaler.inverse_transform(y_hat)
    # y_hat_reshape=y_hat.reshape(-1,8)
    mae, rmse, mape, rrse, corr = All_Metrics(y_hat, Y_truth_test, None, None)
    # 重组
    mae_list.append(mae)
    rmse_list.append(rmse)
    # mape_list.append(mape)
    rrse_list.append(rrse)
    corr_list.append(corr)

    Y_truth_test1 = sum(Y_truth_test[:, -1:])
    y_hat1 = sum(y_hat[:, -1:])
    # aa=sum(Y_truth_test1)
    # bb=sum(y_hat1)
    aa = Y_truth_test1
    bb = y_hat1
    acc1 = abs(bb[0] - aa[0]) / aa[0] * 100
    # acc2 = abs(bb[1] - aa[1]) / aa[1] * 100
    # acc3 = abs(bb[2] - aa[2]) / aa[2] * 100
    # acc4 = abs(bb[3] - aa[3]) / aa[3] * 100
    # acc = (acc1 + acc2 + acc3 + acc4) / 4
    acc = acc1
    if acc < 5:
        line = line + 1
    if acc < 3:
        line1 = line1 + 1
    if acc < 2:
        line2 = line2 + 1
    acc_num.append(acc)

if i//11==0 and i<240:
        np.savetxt(f'pre/pre_{i}.csv', y_hat, delimiter=',', fmt='%.4f')
        np.savetxt(f'true/true_{i}.csv', Y_truth_test, delimiter=',', fmt='%.4f')
print(line/(test_X.shape[0] // batch_size))
mae = np.mean(np.array(mae_list))
rmse = np.mean(np.array(rmse_list))
# mape = np.mean(np.array(mape_list))
rrse = np.mean(np.array(rrse_list))
corr = np.mean(np.array(corr_list))
line=100*line/test_X.shape[0]
line1=100*line1/test_X.shape[0]
line2=100*line2/test_X.shape[0]
print("预测误差小于5%准确百分比:",line)
print("预测误差小于3%准确百分比:",line1)
print("预测误差小于2%准确百分比:",line2)
print('MAE =', '{:.6f}'.format(mae), 'RMSE =', '{:.6f}'.format(rmse),'RRSE =', '{:.6f}'.format(rrse),'CORR =', '{:.6f}'.format(corr))

# retrain
data_all = pd.read_csv('data/MM-2-notime.csv', header=None)
data_all = np.array(data_all)
# 归一化
minimum = data_all.min()
maximum = data_all.max()
print("maximum:",maximum)
print("minimum:",minimum)
scaler = MinMax01Scaler(minimum, maximum)
data_all = scaler.transform(data_all)
print(data_all.shape)

data_train = data_all[:int(len(data_all) * 0.8), :]
# data_train = data_all[:1280,:]
print(data_train.shape)

time_step=12
pre_step=1

train_X, train_Y = create_dataset(data_train, time_step, pre_step)

print(train_X.shape)
print(train_Y.shape)
new_model.compile(loss='mse', optimizer='adam')
new_model.fit(train_X, train_Y, epochs=50, batch_size=64, verbose=1 )#,callbacks=[cp_callback])
new_model.summary()
new_model.save('savemodel/transpre_model'+str(time_step)+'-'+str(pre_step)+'.h5')
data_test = data_all[int(len(data_all) * 0.8):, :]

print(data_test.shape)

# print(y_hat.dtype)

test_X, Y_truth = create_dataset(data_test, time_step, pre_step)

# print(test_X.shape)
# print(Y_truth.shape)
# Y_pred = []
mae_list = []
rmse_list = []
mape_list = []
rrse_list = []
corr_list = []
test_X_test = []
acc_num=[]
true_n=[]
pre_n=[]
line=0
line1=0
line2=0
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
    # test_X_test = np.swapaxes(test_X_test, 1, 0)
    Y_truth_test = np.array(Y_truth_test)
    Y_truth_test = Y_truth_test.reshape(y_hat.shape[0], -1)

    Y_truth_test = scaler.inverse_transform(Y_truth_test)
    y_hat = scaler.inverse_transform(y_hat)
    # y_hat_reshape=y_hat.reshape(-1,8)
    mae, rmse, mape, rrse, corr = All_Metrics(y_hat, Y_truth_test, None, None)

    # 重组
    Y_truth_test1 = sum(Y_truth_test[:, -1:])
    y_hat1 = sum(y_hat[:, -1:])
    mae_list.append(mae)
    rmse_list.append(rmse)
    mape_list.append(mape)
    rrse_list.append(rrse)
    corr_list.append(corr)
    aa = Y_truth_test1
    bb = y_hat1
    acc1 = abs(bb[0] - aa[0]) / aa[0] * 100
    # acc2 = abs(bb[1] - aa[1]) / aa[1] * 100
    # acc3 = abs(bb[2] - aa[2]) / aa[2] * 100
    # acc4 = abs(bb[3] - aa[3]) / aa[3] * 100
    # acc = (acc1 + acc2 + acc3 + acc4) / 4
    acc = acc1
    pre_n.append(aa)
    true_n.append(bb)
    acc_num.append(acc)
    if acc<5:
        line=line+1
    if acc<3:
        line1=line1+1
    if acc<2:
        line2=line2+1
    if i//11==0 and i<240:
        np.savetxt(f'pre/pre_{i}.csv', y_hat, delimiter=',', fmt='%.4f')
        np.savetxt(f'true/true_{i}.csv', Y_truth_test, delimiter=',', fmt='%.4f')
line=100*line/test_X.shape[0]
line1=100*line1/test_X.shape[0]
line2=100*line2/test_X.shape[0]
print("预测误差小于5%准确百分比:",line)
print("预测误差小于3%准确百分比:",line1)
print("预测误差小于2%准确百分比:",line2)
print("最大误差率为：",max(acc_num))
print("最小误差率为：",min(acc_num))
print("平均误差率：",np.mean(acc_num))
np.savetxt(f'each-accnumber.csv',acc_num,delimiter=',',fmt='%.4f')
mae = np.mean(np.array(mae_list))
rmse = np.mean(np.array(rmse_list))
mape = np.mean(np.array(mape_list))
rrse = np.mean(np.array(rrse_list))
corr = np.mean(np.array(corr_list))
print('MAE =', '{:.6f}'.format(mae), 'RMSE =', '{:.6f}'.format(rmse),'RRSE =', '{:.6f}'.format(rrse),'CORR =', '{:.6f}'.format(corr))
plt.plot(pre_n,color="#8ECFC9", label="ture")
plt.plot(true_n,color="coral", label="pre")
plt.legend()
plt.xlabel("time_step")  # x轴命名表示
plt.ylabel("temperature")  # y轴命名表示
plt.title('Comparison of Predicted and True Values')
plt.show()

