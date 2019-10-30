# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 22:32:09 2018

@author: 85402
"""

import numpy as np
import pandas as pd
train_data = pd.read_excel(r"C:\Users\85402\Desktop\台大机器学习编程作业\hw1 Q18\hw1 Q18 train_data.xlsx",header=None)
test_data = pd.read_excel(r"C:\Users\85402\Desktop\台大机器学习编程作业\hw1 Q18\hw1 Q18 test_data.xlsx",header=None)
train_data.columns = ['x1','x2','x3','x4','y']
test_data.columns = ['x1','x2','x3','x4','y']
train_data_x = train_data.drop(['y'],axis=1)
train_data_y = train_data.y
test_data_x = test_data.drop(['y'],axis=1)
test_data_y = test_data.y


def init():
    W = np.zeros((1,4))
    b = 0
    return W,b



def PLA_50(train_data_x,train_data_y,test_data_x,test_data_y):
        W,b = init()
        sample = np.random.randint(len(train_data_x), size=len(train_data_x))
        count = 0
        for j in range(5):
            for i in sample:
                outcome = np.matmul(W,train_data_x.values[i].reshape(4,1)) + b
                if outcome * train_data_y.values[i] <=0:
                    W = W + train_data_y.values[i]*train_data_x.values[i]
                    b = b+ train_data_y.values[i]
                    count +=1
                if count == 51:
                    return W,b





def verify(train_data_x,train_data_y,test_data_x,test_data_y):
    all_error = 0
    for z in range(2000):
        W,b = PLA_50(train_data_x,train_data_y,test_data_x,test_data_y)
        sample = np.random.randint(len(test_data_x), size=len(test_data_x))
        error = 0
        for i in sample: 
            outcome = np.matmul(W,test_data_x.values[i].reshape(4,1)) + b
            if outcome*test_data_y[i] <=0:
                error += 1
        all_error += error
    all_error = all_error/2000/500
    return all_error


a = verify(train_data_x,train_data_y,test_data_x,test_data_y)
print(a)