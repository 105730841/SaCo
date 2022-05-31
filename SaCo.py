# coding: utf-8 -*-
"""
Created on Thu Aug 17 20:45:30 2021
SaCo
@author: hp
"""

import torch
import os
import random as rd
import numpy as np
import cvxopt
import pandas as pd
import Second_contrainer
import copy
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error as mse
import warnings
import math
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1234)

warnings.filterwarnings("ignore")


class Net1(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_hidden1, n_hidden2,
                 n_output):
        super(Net1, self).__init__()

        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_feature + n_hidden2, n_output)

    def forward(self, x):
        inputs = x
        x = self.hidden(x)
        x = F.relu(x)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = torch.cat([inputs, x], dim=-1)

        x = self.predict(x)

        return x


class Net2(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_hidden1, n_hidden2,
                 n_output):
        super(Net2, self).__init__()

        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)

        x = self.predict(x)
        return x


def train_iteration(model, data, label, num_epoch, learning_rate):
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).float()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()

    for e in range(num_epoch):
        label_pre = model(data)
        loss = loss_func(label_pre, label)
        if (e % 100 == 0):
            print(f'Epoch:{e},loss:{loss.data.numpy()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def main():
    start = time.perf_counter()
    path1 = os.getcwd() + '\data'
    data = ['cpu_small']

    for i in range(len(data)):
        dataname = data[i]
        if i > 1:
            lr1 = 0.01
        else:
            lr1 = 0.1

        for j in range(4):
            # Different label size setting
            label_index = 50 * (2 ** j)
            file_number = 20

            log = np.zeros((1, 11))
            rmse_result = np.zeros(file_number)

            r2_result = np.zeros(file_number)

            for k in range(file_number):
                print(k)
                path = path1 + '\\%s' % (dataname) + '.arff_{}.csv'.format(k)
                f = open(path, encoding='utf-8')

                train_data, train_label, unlabel_data, unlabel_label, test_data, test_label, num_features = read_data(f,
                                                                                                                      label_index,
                                                                                                                      2000)
                net1 = Net1(n_feature=num_features, n_hidden=32, n_hidden1=64, n_hidden2=32, n_output=1)
                net2 = Net2(n_feature=num_features, n_hidden=64, n_hidden1=64, n_hidden2=64, n_output=1)

                # Semi-supervised regressors
                net3 = train_iteration(net1, train_data, train_label, 3000, lr1)
                net4 = train_iteration(net2, train_data, train_label, 3000, lr1)

                # Baselines
                net5 = train_iteration(net1, train_data, train_label, 3000, lr1)
                net6 = train_iteration(net2, train_data, train_label, 3000, lr1)
                if label_index > 200 and i != 2:
                    lr1 = 0.1

                # Train
                net3, net4, log_mse, iternumber = saco(net3, net4, net5, net6, 3, train_data, train_label, 3,
                                                       unlabel_data, test_data, test_label, 100, 1, 100, lr1)

                # Record the mse, and finally output the log file
                log = np.concatenate((log, log_mse), axis=0)

                # Prediction
                temp_train_data = torch.from_numpy(train_data)
                train_label_pre1 = net3(temp_train_data).data.numpy()
                train_label_pre2 = net4(temp_train_data).data.numpy()
                mse1 = mse(train_label_pre1, train_label)
                mse2 = mse(train_label_pre2, train_label)
                weight1 = math.pow(1 / mse1, 3) / (math.pow(1 / mse1, 3) + math.pow(1 / mse2, 3))
                weight2 = 1 - weight1

                temp_test_data = torch.from_numpy(test_data)
                test_pre1 = net3(temp_test_data).data.numpy()
                test_pre2 = net4(temp_test_data).data.numpy()
                temp_pre = test_pre1 * weight1 + test_pre2 * weight2

                rmse_result[k] = math.sqrt(mse(test_label, temp_pre))

                r2_result[k] = R_2(test_label, temp_pre)  # r2 of model

            log = log[1:log.shape[0], :]
            log_dt = pd.DataFrame(log)
            logname = dataname + '_{}'.format(label_index)
            log_dt.to_csv(os.getcwd() + '\log\\train_log' + '\\' + logname + '.csv', index=False)

            rmseresult_mean = rmse_result.mean()
            rmseresult_std = rmse_result.std()
            all_rmse = np.array(['rmse', rmseresult_mean, rmseresult_std]).reshape(-1, 1)

            r2result_mean = r2_result.mean()
            r2result_std = r2_result.std()
            all_r2 = np.array(['r2', r2result_mean, r2result_std]).reshape(-1, 1)

            total_result = np.concatenate((all_rmse, all_r2), axis=1)
            total_result_dt = pd.DataFrame(total_result)
            resultname = dataname + '_total_{}'.format(label_index)
            total_result_dt.to_csv(os.getcwd() + '\log\\total' + '\\' + resultname + '.csv', index=False)

    end = time.perf_counter()
    print('runtime:', end - start)
    return 0


def saco(regressor1, regressor2, base_regressor1, base_regressor2, kValue1, train_data, train_label, kValue2,
         unlabel_data, test_data, test_label, poolSize, paraThreshold, train_iterations, learning_rate):
    data1 = copy.deepcopy(train_data)
    label1 = copy.deepcopy(train_label)
    data2 = copy.deepcopy(train_data)
    label2 = copy.deepcopy(train_label)

    baseline1 = copy.deepcopy(base_regressor1)
    baseline2 = copy.deepcopy(base_regressor2)

    traindata1 = copy.deepcopy(train_data)
    trainlabel1 = copy.deepcopy(train_label)
    traindata2 = copy.deepcopy(train_data)
    trainlabel2 = copy.deepcopy(train_label)

    all_add_data1 = np.empty(shape=[0, train_data.shape[1]])
    all_add_data2 = np.empty(shape=[0, train_data.shape[1]])

    log_mse = np.zeros((1, int((train_iterations / 10) + 1)))
    cnt = 0

    for i in range(train_iterations):
        # Select for view2
        flag2, add_data2, add_label2, delete_index = Second_contrainer.net_selectCriticalInstances_knn(regressor1,
                                                                                                       baseline1,
                                                                                                       kValue1, data1,
                                                                                                       label1,
                                                                                                       unlabel_data,
                                                                                                       poolSize,
                                                                                                       learning_rate)
        if flag2 == 1:
            all_add_data2 = np.concatenate((all_add_data2, add_data2), axis=0)
            if i != 0:
                first_add_data2 = torch.from_numpy(all_add_data2).float()
                first_add_label2 = regressor2(first_add_data2).data.numpy()
                first_add_label = regressor1(first_add_data2).data.numpy()
                semi_label2 = np.concatenate((first_add_label, first_add_label2), axis=1)
                base_label2_1 = base_regressor1(first_add_data2).data.numpy()
                base_label2_2 = base_regressor2(first_add_data2).data.numpy()
                base_label2 = (base_label2_1 + base_label2_2) / 2
                temp_label2 = safe_labeling(semi_label2, base_label2)
                add_label2 = temp_label2
            else:
                first_add_data2 = torch.from_numpy(add_data2).float()
                first_add_label2 = regressor2(first_add_data2).data.numpy()
                add_label2 = (add_label2 + first_add_label2) / 2

            # Update view2
            traindata2 = np.concatenate((data2, all_add_data2), axis=0)
            trainlabel2 = np.concatenate((label2, add_label2), axis=0)

            regressor2 = train_iteration(regressor2, traindata2, trainlabel2, 100, learning_rate)

            for j, tempadd_index in enumerate(delete_index):
                unlabel_data = np.delete(unlabel_data, tempadd_index, axis=0)

        # Select for view1
        flag1, add_data1, add_label1, delete_index = Second_contrainer.net_selectCriticalInstances_knn(regressor2,
                                                                                                       baseline2,
                                                                                                       kValue2, data2,
                                                                                                       label2,
                                                                                                       unlabel_data,
                                                                                                       poolSize,
                                                                                                       learning_rate)
        if flag1 == 1:
            all_add_data1 = np.concatenate((all_add_data1, add_data1), axis=0)
            if i != 0:
                second_add_data1 = torch.from_numpy(all_add_data1).float()
                second_add_label1 = regressor1(second_add_data1).data.numpy()
                second_add_label = regressor2(second_add_data1).data.numpy()
                semi_label1 = np.concatenate((second_add_label, second_add_label1), axis=1)
                base_label1_1 = base_regressor1(second_add_data1).data.numpy()
                base_label1_2 = base_regressor2(second_add_data1).data.numpy()
                base_label1 = (base_label1_1 + base_label1_2) / 2
                temp_label1 = safe_labeling(semi_label1, base_label1)
                add_label1 = temp_label1
            else:
                second_add_data1 = torch.from_numpy(add_data1).float()
                first_add_label1 = regressor1(second_add_data1).data.numpy()
                add_label1 = (add_label1 + first_add_label1) / 2

            # Update view2
            traindata1 = np.concatenate((data1, all_add_data1), axis=0)
            trainlabel1 = np.concatenate((label1, add_label1), axis=0)

            regressor1 = train_iteration(regressor1, traindata1, trainlabel1, 100, learning_rate)

            for j, tempadd_index in enumerate(delete_index):
                unlabel_data = np.delete(unlabel_data, tempadd_index, axis=0)

        if flag1 == 0 and flag2 == 0:
            break
        else:
            cnt += 1

        if i % 10 == 0:
            log_mse[0][int(i / 10)] = regressor_mse(regressor1, regressor2, test_data, test_label, 0)

    regressor1 = train_iteration(regressor1, traindata1, trainlabel1, 1000, learning_rate)
    regressor2 = train_iteration(regressor2, traindata2, trainlabel2, 1000, learning_rate)

    log_mse[0][log_mse.shape[1] - 1] = regressor_mse(regressor1, regressor2, test_data, test_label, 0)

    return regressor1, regressor2, log_mse, cnt


def regressor_mse(regressor1, regressor2, test_data, test_label, mode):
    temp_testdata = torch.from_numpy(test_data)
    temp_pre1 = regressor1(temp_testdata)
    temp_pre1 = temp_pre1.data.numpy()
    if mode == 0:
        temp_pre2 = regressor2(temp_testdata)
        temp_pre2 = temp_pre2.data.numpy()
    else:
        temp_pre2 = regressor2.predict(test_data).reshape(-1, 1)

    temp_pre = (temp_pre1 + temp_pre2) / 2
    result_mse = mse(temp_pre, test_label)

    return result_mse


def read_data(path, label_Index, unlabel_Index):
    data = pd.read_csv(path)
    all_features = data.iloc[:, 0:data.shape[1] - 1]
    all_labels = data.iloc[:, data.shape[1] - 1:data.shape[1]]
    all_features = all_features.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    all_labels = all_labels.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    num_index = all_features.shape[0]
    num_features = all_features.shape[1]
    data = all_features[0:num_index].values.astype(np.float32)
    label = all_labels[0:num_index].values.astype(np.float32)
    train_data = data[0:label_Index, :]
    train_label = label[0:label_Index, :]
    unlabel_data = data[label_Index:unlabel_Index, :]
    unlabel_label = label[label_Index:unlabel_Index, :]
    test_data = data[unlabel_Index:data.shape[0], :]
    test_label = label[unlabel_Index:label.shape[0], :]
    return train_data, train_label, unlabel_data, unlabel_label, test_data, test_label, num_features


def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert (k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert (Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq, lb, ub)

    return np.array(sol['x'])


def safe_labeling(candidate_prediction, baseline_prediction):
    semi_pre = copy.deepcopy(candidate_prediction.astype(np.float64))
    supervised_pre = copy.deepcopy(baseline_prediction)
    prediction_num = candidate_prediction.shape[1]
    H = np.dot(semi_pre.T, semi_pre) * 2
    f = -2 * np.dot(semi_pre.T, supervised_pre)
    Aeq = np.ones((1, prediction_num))
    beq = 1.0
    lb = np.zeros((prediction_num, 1))
    ub = np.ones((prediction_num, 1))
    sln = quadprog(H, f, None, None, Aeq, beq, )
    safer_prediction = np.zeros((semi_pre.shape[0], 1))
    for i in range(safer_prediction.shape[0]):
        tempsafer = 0
        for j in range(prediction_num):
            tempsafer = tempsafer + sln[j] * semi_pre[i, j]
        safer_prediction[i][0] = tempsafer
    return safer_prediction


def R_2(label, predict_label):
    r_2 = 0
    label = label.reshape(-1, 1)
    predict_label = predict_label.reshape(-1, 1)
    result = {}
    ybar = np.sum(label) / label.shape[0]
    ssreg = np.sum((predict_label - label) ** 2)
    sstot = np.sum((label - ybar) ** 2)
    r_2 = 1 - ssreg / sstot
    result['R^2'] = r_2
    return r_2


if __name__ == '__main__':
    main()
