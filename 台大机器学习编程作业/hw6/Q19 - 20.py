# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:25:05 2019

@author: 习近平
"""


import pandas as pd
import numpy as np
from math import e


all_data = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw6\hw6_lssvm_all.dat.xlsx',header = None)


all_data.columns = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','y']
train = all_data.loc[0:399,['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','y']]
test = all_data.loc[400:,['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','y']]


train_x = train[['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9']].values.reshape(len(train),len(train.columns)-1)
test_x = test[['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9']].values.reshape(len(test),len(test.columns)-1)
train_y = train.y.values.reshape(len(train),1)
test_y = test.y.values.reshape(len(test),1)



def RBF_kernel(gamma,x1,x2):
    K = np.zeros((len(x1),len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            K[i][j] = e**(-gamma*(np.sum((x1[i]-x2[j])**2)))    
    return K

def get_beta(K,gamma,lamda,x1,x2,y):     
    beta = np.matmul(np.linalg.inv(lamda * np.identity(len(K)) + K),y)
    return beta


for gamma in [32,2,0.125]:
    for lamda in [0.001,1,1000]:
        print('Ein:','-'*25)
        K = RBF_kernel(gamma,train_x,train_x)
        beta = get_beta(K,gamma,lamda,train_x,train_x,train_y)          #beta 等于 W* Zn ,所以最后得出的beta是跟w一样的shape，再跟每一个样本点相乘就得出了pre
        pre = np.dot(K.T,beta)
        pre = np.array(np.sign(pre))
        print('param:{},{}'.format(gamma,lamda))
        print('Ein:',sum(train_y != pre)/400)
        print('Eout:','-'*25)
        K2 = RBF_kernel(gamma,train_x,test_x)
        test_pre = np.dot(K2.T,beta)
        test_pre = np.array(np.sign(test_pre))
        print('Eout:',sum(test_y != test_pre)/100)
        