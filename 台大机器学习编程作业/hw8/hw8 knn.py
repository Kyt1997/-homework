# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:14:16 2019

@author: 习近平
"""

import pandas as pd
import numpy as np

train  = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw8\hw8 knn train.xlsx',header = None)
test = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw8\hw8 knn test.xlsx',header = None)
train.columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','y']
test.columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','y']

'''train_x = train[['x1','x2','x3','x4','x5','x6','x7','x8','x9']].values.reshape(len(train),9)
train_y = train.y.values.reshape(len(train),1)
'''

def predict(data1,data2,k):
    data1_x = data1[['x1','x2','x3','x4','x5','x6','x7','x8','x9']].values.reshape(len(data1),9)
    data1_y = data1.y.values.reshape(len(data1),1)
    data2_x = data2[['x1','x2','x3','x4','x5','x6','x7','x8','x9']].values.reshape(len(data2),9)
    data2_y = data2.y.values.reshape(len(data2),1)
    gt = []
    temp = []

    for i in range(len(data2)):
        temp = []
        pre = 0
        for j in range(len(data1)):
            distance = np.linalg.norm(data2_x[i] - data1_x[j])
            temp.append((distance,data1_y[j]))
        temp = sorted(temp,key=lambda x:x[0])
        best_y = temp[:k]
        for c in range(k):
            pre += best_y[c][1]
                           
        gt.append(np.sign(pre))

    return gt,data2_y

def error(gt,data2_y):
    gt = np.array(gt).reshape(len(gt),1)
    E = sum(gt != data2_y) / len(data2_y)
    #print(gt)
    return E

def run(data1,data2,k):
    gt,data2_y = predict(data1,data2,k)
    E = error(gt,data2_y)
    #print(E)
    return E

run(train,train,5)
run(train,test,5)