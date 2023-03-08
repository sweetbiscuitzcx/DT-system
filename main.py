# 进行LSTM多维时间序列预测
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
import matplotlib as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
tf.compat.v1.Session(config=config)


def create_dataset(data, n_predictions, n_next):
    '''
    对数据进行处理
    '''
    dim = data.shape[1]
    train_X, train_Y = [], []
    for i in range(data.shape[0] - n_predictions - n_next - 1):
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


def trainModel(train_X, train_Y):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = Sequential()
    model.add(LSTM(
        140,
        input_shape=(train_X.shape[1], train_X.shape[2]),
        return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(
        140,
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        train_Y.shape[1]))
    model.add(Activation("relu"))

    # model.compile(loss='mse', optimizer='adam')

    # checkpointPath = "training_1/cp.ckpt"
    # checkpointDir = os.path.dirname(checkpointPath)
    #
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)

    # model.fit(train_X, train_Y, epochs=100, batch_size=64, verbose=1 )#,callbacks=[cp_callback])
    model.summary()



    return model


# def reshape_y_hat(y_hat, dim):
#     re_y = []
#     i = 0
#     while i < len(y_hat):
#         tmp = []
#         for j in range(dim):
#             tmp.append(y_hat[i + j])
#         i = i + dim
#         re_y.append(tmp)
#     re_y = np.array(re_y, dtype='float64')
#     return re_y

# train
from metrics import All_Metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from normalization import MinMax01Scaler
import pandas as pd

data_all = pd.read_csv('data/MM-1-notime.csv', header=None)
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


model = trainModel(train_X, train_Y)
model.compile(loss='mse', optimizer='adam')
model.fit(train_X, train_Y, epochs=50, batch_size=64, verbose=1 )#,callbacks=[cp_callback])
# model.save_weights('savemodel/modelfinish')
model.save('savemodel/my_model'+str(time_step)+'-'+str(pre_step)+'.h5')

# test
batch_size =1

# 仅对最后200条数据进行测试 因为预测仅最新有作用
# data_test = data_all[int(len(data_all)*0.7):,:]
data_test = data_all[int(len(data_all) * 0.5):, :]

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
true_n=[]
test_X_test = []
acc_num=[]
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
    y_hat = model.predict(X)
    # print(y_hat.shape)
    y_hat = y_hat.reshape(pre_step, -1)#此处做修改，原来为batch——size，现为所预测步数
    # print(y_hat.shape)
    '''
    for i in [0,2,3,8,11,12,13,14,18,19,20,21,23,24,25,27]:
        test_X_test.append(X[:,:,i]) 
    '''
    # test_X_test.append(X[:,:,:])
    # test_X_test = np.swapaxes(test_X_test, 1, 0)
    # test_X_test = np.array(test_X_test)
    # test_X_test = test_X_test.reshape(y_hat.shape[0],y_hat.shape[1],y_hat.shape[2])
    Y_truth_test.append(Y[:, :])
    # test_X_test = np.swapaxes(test_X_test, 1, 0)
    Y_truth_test = np.array(Y_truth_test)
    Y_truth_test = Y_truth_test.reshape(y_hat.shape[0], -1)

    Y_truth_test = scaler.inverse_transform(Y_truth_test)
    y_hat = scaler.inverse_transform(y_hat)
    # y_hat_reshape=y_hat.reshape(-1,8)
    mae, rmse, _, rrse, corr = All_Metrics(y_hat, Y_truth_test, None, None)
    # np.savetxt(f'pre/pre_{i}.csv', y_pre, delimiter=',', fmt='%.4f')
    # np.savetxt(f'pre/true_{i}.csv', y_true, delimiter=',', fmt='%.4f')
    # test_X_test.append(X[:,:,i])
    # Y_pred.append(y_hat)
    # 重组
    Y_truth_test1 = sum(Y_truth_test[:, -1:])
    y_hat1 = sum(y_hat[:, -1:])
    mae_list.append(mae)
    rmse_list.append(rmse)
    # mape_list.append(mape)
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
np.savetxt(f'数据/MM-1-notime/pre_.csv', pre_n, delimiter=',', fmt='%.4f')
np.savetxt(f'数据/MM-1-notime/true_.csv', true_n, delimiter=',', fmt='%.4f')
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
# mape = np.mean(np.array(mape_list))
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