# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:30:06 2019

@author: 85402
"""




import pandas as pd
import numpy as np
import math

train = pd.read_excel(r"C:\Users\85402\Desktop\台大机器学习编程作业\hw3 Q19&20\hw3 Q19&20 train data.xlsx",header=None)
test = pd.read_excel(r"C:\Users\85402\Desktop\台大机器学习编程作业\hw3 Q19&20\hw3 Q19&20 test_data.xlsx",header=None)

train.columns = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','y']
test.columns = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','y']


train_x = train[['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20']]
train_y = train['y']
#以下是Q18 & Q19 batch gradient descent
train_y = np.array(train_y).reshape(1000,1)
train_x = np.array(train_x).reshape(1000,21)
w = np.zeros((21,1))
learning_rate = 0.001
T = 2000


gradient_Ein = 0
for t in range(T):   
    print("这是第",t,"轮")
    gradient_Ein = sum(np.multiply((1/(1 + math.e**(np.multiply(train_y,np.matmul(train_x,w))))) , np.multiply(-train_y,train_x)))/1000
    w -= learning_rate * gradient_Ein.reshape(21,1)
print(w)
   
  
#下面为Eout  
test_x = test[['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20']]
test_y = test['y']
test_y = np.array(test_y)
predict = []

for c in range(3000):
    pre = 1/(1+math.e**(-np.matmul(test_x.values[c],w)))
    if pre >= 0.5:
        predict.append(1)
    else:
        predict.append(-1)

predict = np.array(predict)

sum(predict != test_y)/3000        
    


       

   
              