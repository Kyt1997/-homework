# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:27:45 2019

@author: 85402
"""
import numpy as np
import pandas as pd
train = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw4 Q13--20\hw4 Q13 train.xlsx',header=None)
test = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw4 Q13--20\hw4 Q13 test.xlsx',header=None)
train.columns = ['x1','x2','y']
test.columns = ['x1','x2','y']
train['x0'] = 1
test['x0'] = 1
train_x = np.array(train[['x0','x1','x2']])
train_y = np.array(train['y'])
test_y = np.array(test['y'])
test_x = np.array(test[['x0','x1','x2']])


#通过调节lamda来完成14题
lamda = 10**(-3)
"""由hw4的Q12可以知道 有正则化的 w_reg = (X.T * y) / (lambda + X.T *X)"""
#下面进行13题Ein的测量
I = np.eye(3,3)
W_reg = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_x.T,train_x) + lamda*I),train_x.T),train_y.reshape(200,1))

W_reg = W_reg.reshape(3,1)

Ein = []
for i in range(200):
    c = np.matmul(train_x[i].reshape(1,3),W_reg)
    
    if c[0][0] > 0:
        Ein.append(1)
    else:
        Ein.append(-1)
   
Ein = np.array(Ein).reshape(200,1)


train_y = train_y.reshape(200,1)

print('在当前的lambda: ',lamda,'下，Ein是: ',sum(Ein != train_y)/200)


#下面进行13题目Eout的测量
Eout = []
for i in range(1000):
    a = np.matmul(test_x[i].reshape(1,3),W_reg)
    if a > 0:
        Eout.append(1)
    else:
        Eout.append(-1)

Eout = np.array(Eout).reshape(1000,1)
test_y = test_y.reshape(1000,1)
print('在当前的lambda: ',lamda,'下，Eout是: ',sum(test_y != Eout)/1000)
