# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import copy
from sklearn.model_selection import train_test_split
import os



def EuclideanDistance(x, y):
    """
    get the Euclidean Distance between to matrix
    (x-y)^2 = x^2 + y^2 - 2xy
    :param x:
    :param y:
    :return:
    """
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis

def predict(kValue,train_data,train_label,paraInstance):
    tempNeighbor=findneighbor(kValue,train_data,train_label,paraInstance)
    tempPre=sum(train_label[idx] for i,idx in enumerate(tempNeighbor))
    Instance_predict=tempPre/kValue
    return Instance_predict

def predict_data(kValue,train_data,train_label,paradata):
    prelabel = np.zeros((1,1))
    for i, paraInstance in enumerate(paradata):
        paraInstance = paraInstance.reshape(1,-1)
        tempNeighbor=findneighbor(kValue,train_data,train_label,paraInstance)
        tempPre=sum(train_label[idx] for i,idx in enumerate(tempNeighbor))
        Instance_predict=tempPre/kValue
        Instance_predict = Instance_predict
        prelabel = np.concatenate((prelabel,Instance_predict),axis=0)

    prelabel = prelabel[1:prelabel.shape[0]]
    return prelabel


def selfpredict(kValue,train_data,train_label,paraInstance):
    tempNeighbor=findSelfneighbor(kValue,train_data,train_label,paraInstance)
    tempPre=sum(train_label[idx] for i,idx in enumerate(tempNeighbor))
    Instance_predict=tempPre/kValue
    return Instance_predict

def selfpredict1(kValue,train_data,train_label,paraInstance):
    tempNeighbor=findSelfneighbor(kValue,train_data,train_label,paraInstance)
    distance = []
    sum_weight = 0
    Instance_predict = 0
    for i, idx in enumerate(tempNeighbor):
        distance.append(EuclideanDistance(train_data[idx], paraInstance))
        sum_weight += 1/distance[i]
    for i, idx in enumerate(tempNeighbor):
        Instance_predict += train_label[idx] * ((1/distance[i])/sum_weight)
    # tempPre=sum(train_label[idx] for i,idx in enumerate(tempNeighbor))
    # Instance_predict=tempPre/kValue
    return Instance_predict

def selfpredict3(kValue,train_data,train_label,paraInstance):
    tempNeighbor=findSelfneighbor(kValue,train_data,train_label,paraInstance)
    distance = []
    sum_weight = 0
    Instance_predict = 0
    for i, idx in enumerate(tempNeighbor):
        distance.append(EuclideanDistance(train_data[idx], paraInstance))
        sum_weight += math.pow(1/distance[i], 3)
    for i, idx in enumerate(tempNeighbor):
        Instance_predict += train_label[idx] * ((math.pow(1/distance[i], 3))/sum_weight)
    # tempPre=sum(train_label[idx] for i,idx in enumerate(tempNeighbor))
    # Instance_predict=tempPre/kValue
    return Instance_predict

def findneighbor(kValue,train_data,train_label,paraInstance):
    tempIndex=[]
    resultIndex=[]
    returnIndex=[]
    tempIndex=EuclideanDistance(train_data, paraInstance)
    resultIndex=np.argsort(tempIndex,0)
    returnIndex=resultIndex[0:kValue:1]  
    return returnIndex

def findSelfneighbor(kValue,train_data,train_label,paraInstance):
    tempIndex=[]
    resultIndex=[]
    returnIndex=[]
    tempIndex=EuclideanDistance(train_data, paraInstance)
    resultIndex=np.argsort(tempIndex,0)
    returnIndex=resultIndex[1:kValue+1:1]   
    return returnIndex


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
    # train_Size=int (0.05*len(data))
    # tests_Size=int (0.3*len(data))
    train_data, unuse_data, train_label, unuse_label = train_test_split(data, label, test_size=0.99)
    unlabel_data, test_data, unlabel_label, test_label = train_test_split(unuse_data, unuse_label,
                                                                          test_size=(0.3) / (1 - 0.01))
    mse = 0
    # for i, paraInstance in enumerate(train_data):
    #     paraInstance = paraInstance.reshape(1, -1)
    #     paraLabel = predict(3, train_data, train_label, paraInstance)
    #     # print(train_data.shape[1])
    #     neighborLabel = selfpredict(3, train_data, train_label, paraInstance)
    #     # neighborLabel2=findSelfneighbor(3,train_data,train_label,paraInstance,i)
    #     print(neighborLabel)
    #     # print(neighborLabel2)
    #     mse += (paraLabel - unlabel_label[i]) * (paraLabel - unlabel_label[i])
    #
    # print(mse / unlabel_data.shape[0])
    paradata = copy.deepcopy(train_data)
    print('trainlabel',train_label)
    pre = predict_data(3,train_data,train_label,paradata)
    print('pre',pre)

if __name__ == '__main__':
    main()
