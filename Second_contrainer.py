import pandas as pd
import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import mean_squared_error as mse
import copy
import KNNRegressor
import random
import random as rd

rd.seed(1234)


def select_train_iteration(model, data, label, num_epoch, learning_rate):
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).float()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()

    for e in range(num_epoch):
        label_pre = model(data)
        loss = loss_func(label_pre, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def net_selectCriticalInstances_knn(regressor1, regressor2, kValue, train_data, train_label, unlabel_data, poolsize,
                                    learning_rate):
    if unlabel_data.shape[0] < poolsize:
        tempPool = np.arange(unlabel_data.shape[0])
    elif unlabel_data.shape[0] >= poolsize:
        tempPool = random.sample(range(0, unlabel_data.shape[0]), poolsize)
    nodata = -1
    nolabel = -1
    noindex = -1

    flag = 1
    select_number = 5
    n_learners = 2
    delta = []
    tempValue = []
    paradata = copy.deepcopy(train_data)
    label_pre = KNNRegressor.predict_data(3, train_data, train_label, paradata)
    difference = train_label - label_pre
    # Calculate delta
    for i, paraUnlabel in enumerate(tempPool):
        tempInstance = copy.deepcopy(unlabel_data[paraUnlabel])
        tempInstance = tempInstance.reshape(1, -1)
        tempnetInstance = torch.from_numpy(tempInstance).float()
        tempnetLabel = regressor1(tempnetInstance)
        tempLabel = tempnetLabel.data.numpy()

        tempTrain_data = copy.deepcopy(train_data)
        tempTrain_label = copy.deepcopy(train_label)
        tempNeighbor = KNNRegressor.findneighbor(kValue, train_data, train_label, tempInstance)
        insertInstance = np.array(tempInstance)
        tempTrain_data = np.concatenate((tempTrain_data, insertInstance), axis=0)
        tempTrain_label = np.concatenate((tempTrain_label, tempLabel), axis=0)
        tempNewValue = 0
        tempOldValue = 0
        for j, paraNeighbor in enumerate(tempNeighbor):
            tempNeighborInstance = train_data[paraNeighbor]
            tempNeighborInstance = tempNeighborInstance.reshape(1, -1)
            tempOldError = difference[paraNeighbor]
            tempOldValue += tempOldError * tempOldError
            tempNewError = (KNNRegressor.selfpredict(kValue, tempTrain_data, tempTrain_label, tempNeighborInstance) -
                            tempTrain_label[paraNeighbor])
            tempNewValue += tempNewError * tempNewError
        delta.append(tempOldValue / tempNeighbor.shape[0] - tempNewValue / tempNeighbor.shape[0])
        tempValue.append(tempLabel)
    # Select candidates
    delta = np.array(delta).flatten()
    arg1to5 = np.argsort(delta)
    arg1to5 = arg1to5[-1:-(select_number + 1):-1]
    arg1to5 = np.sort(arg1to5)

    # Select the most confident instance
    k = 0
    x_delta = np.zeros((select_number, n_learners))
    for i, paraunlabel in enumerate(tempPool):
        if i in arg1to5:
            x_u = copy.deepcopy(unlabel_data[paraunlabel]).reshape(1, -1)

            oldtraindata = copy.deepcopy(train_data)
            oldtraindata_tensor = torch.from_numpy(oldtraindata).float()
            oldtrainlabel = copy.deepcopy(train_label)
            templearner1 = copy.deepcopy(regressor1)
            templearner2 = copy.deepcopy(regressor2)

            pre_xu = tempValue[i]
            newdata = np.concatenate((oldtraindata, x_u), axis=0)
            newlabel = np.concatenate((oldtrainlabel, pre_xu), axis=0)

            templearner = [templearner1, templearner2]
            for j in range(n_learners):
                oldpre = templearner[j](oldtraindata_tensor).data.numpy()
                oldmse = mse(oldtrainlabel, oldpre)
                templearner[j] = select_train_iteration(templearner[j], newdata, newlabel, 30,
                                                        learning_rate)
                newpre = templearner[j](oldtraindata_tensor).data.numpy()
                newmse = mse(oldtrainlabel, newpre)
                x_delta[k][j] = oldmse - newmse
            k += 1
    sum_x = np.sum(x_delta, axis=1)
    id = np.argmax(sum_x)

    if sum_x[id] < 0:
        flag = 0
        return flag, nodata, nolabel, noindex
    else:
        maxindex = arg1to5[id]

        bestindex = []
        bestdata = []
        bestlabel = []
        for i, paraunlabel in enumerate(tempPool):
            if i == maxindex:
                bestindex.append(paraunlabel)
                bestinstance = copy.deepcopy(unlabel_data[paraunlabel]).reshape(1, -1)
                bestdata.append(bestinstance)
                bestlabel.append(tempValue[i])

        bestdata = np.array(bestdata).reshape(-1, train_data.shape[1])
        bestlabel = np.array(bestlabel).reshape(-1, 1)
        return flag, bestdata, bestlabel, bestindex


def main():
    path = os.getcwd() + '\generatedata\kin8nm.arff_0.csv'
    f = open(path, encoding='utf-8')
    data = pd.read_csv(f)
    all_features = data.iloc[:, 0:data.shape[1] - 1]
    all_labels = data.iloc[:, data.shape[1] - 1:data.shape[1]]
    all_features = all_features.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    all_labels = all_labels.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    num_index = all_features.shape[0]
    data = all_features[0:num_index].values.astype(np.float32)
    label = all_labels[0:num_index].values.astype(np.float32)
    train_data, unuse_data, train_label, unuse_label = train_test_split(data, label, test_size=0.99)
    unlabel_data, test_data, unlabel_label, test_label = train_test_split(unuse_data, unuse_label,
                                                                          test_size=(0.3) / (1 - 0.01))
    mse = 0
    flag1, resultdata, resultLabel, deleteIndex = net_selectCriticalInstances_knn(1, 3, train_data, train_label,
                                                                                  unlabel_data, 100, 0.8)

    print(resultdata, resultLabel, deleteIndex)
    print(mse / unlabel_data.shape[0])


if __name__ == '__main__':
    main()
