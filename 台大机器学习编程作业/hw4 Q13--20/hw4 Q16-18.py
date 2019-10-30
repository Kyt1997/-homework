# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:58:37 2019

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


train_120 = train[:120][['x0','x1','x2','y']]
val_80 = train[120:][['x0','x1','x2','y']]
train_x = train_120[['x0','x1','x2']]
train_x = np.array(train_x)
train_y = train_120['y']
train_y = np.array(train_y)
val_x = val_80[['x0','x1','x2']]
val_x = np.array(val_x)
val_y = val_80['y']
val_y = np.array(val_y)

#通过调节lamda来完成16题
lamda = 10**(0)
"""由hw4的Q12可以知道 有正则化的 w_reg = (X.T * y) / (lambda + X.T *X)"""
I = np.eye(3,3)
W_reg = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_x.T,train_x) + lamda*I),train_x.T),train_y.reshape(120,1))

W_reg = W_reg.reshape(3,1)
#测量Etrain
Etrain = []
for i in range(120):
    c = np.matmul(train_x[i].reshape(1,3),W_reg)
    
    if c[0][0] > 0:
        Etrain.append(1)
    else:
        Etrain.append(-1)

Etrain = np.array(Etrain).reshape(120,1)


train_y = train_y.reshape(120,1)

print('在当前的lambda: ',lamda,'下，Etrain是: ',sum(Etrain != train_y)/120)   

#测量Eval
Eval = []
for i in range(80):
    a = np.matmul(val_x[i].reshape(1,3),W_reg)
    if a > 0:
        Eval.append(1)
    else:
        Eval.append(-1)

Eval = np.array(Eval).reshape(80,1)
val_y = val_y.reshape(80,1)
print('在当前的lambda: ',lamda,'下，Eval是: ',sum(val_y != Eval)/80)





#下面进行18题，用train_120得出的最佳lamda，就是1，之后，再用这个lamda去训练200个train得出新的W_reg去测量Ein和Eout
whole_train = train
whole_test = test
whole_train_x = whole_train[['x0','x1','x2']]
whole_train_x = np.array(whole_train_x)
whole_train_y = whole_train['y']
whole_train_y = np.array(whole_train_y)
whole_test_x = whole_test[['x0','x1','x2']]
whole_test_x = np.array(whole_test_x)
whole_test_y = whole_test['y']
whole_test_y = np.array(whole_test_y)


lamda = 10**(0)
"""由hw4的Q12可以知道 有正则化的 w_reg = (X.T * y) / (lambda + X.T *X)"""
I = np.eye(3,3)
whole_W_reg = np.matmul(np.matmul(np.linalg.inv(np.matmul(whole_train_x.T,whole_train_x) + lamda*I),whole_train_x.T),whole_train_y.reshape(200,1))

whole_W_reg = whole_W_reg.reshape(3,1)



Ein = []
for i in range(200):
    c = np.matmul(whole_train_x[i].reshape(1,3),whole_W_reg)
    
    if c[0][0] > 0:
        Ein.append(1)
    else:
        Ein.append(-1)

Ein = np.array(Ein).reshape(200,1)


whole_train_y = whole_train_y.reshape(200,1)

print('在当前的lambda: ',lamda,'下，Ein是: ',sum(Ein != whole_train_y)/200)

#Eout
Eout = []
for i in range(1000):
    a = np.matmul(whole_test_x[i].reshape(1,3),whole_W_reg)
    if a[0][0] >= 0:
        Eout.append(1)
    else:
        Eout.append(-1)

Eout = np.array(Eout).reshape(1000,1)
whole_test_y = whole_test_y.reshape(1000,1)
print('在当前的lambda: ',lamda,'下，Eout是: ',sum(whole_test_y != Eout)/1000)
    

