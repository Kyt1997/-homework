# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 22:20:00 2018

@author: 85402
"""

import random
import numpy as np
import pandas as pd
data = pd.read_excel(r"C:\Users\85402\Desktop\台大机器学习编程作业\hw1 Q15 PLA\hw1_Q15_dataset.xlsx",header = None)
data.columns = ['x1','x2','x3','x4','y']
X = data.drop(['y'],axis=1)
Y= data.y

def init():
    W = np.zeros((1,4))
    b = 0
    return W,b


def update(train_x,train_y,learning_rate):
    count = 0
    for z in range(2000):       #依照题意重复两千次
        W,b = init()            #每一次实验都要重置参数，否则第一次实验过后用的参数都是已经学习好的
        sample = random.sample(range(len(X)),400)   #已经经过验证确实能产生不重复的数字
        for j in range(3):         
            for i in sample:
                pre_y = np.matmul(W,X.values[i].reshape(4,1)) + b
                if pre_y * Y.values[i] <=0:
                    W = W+learning_rate*Y.values[i]*X.values[i]
                    b = b+learning_rate*Y.values[i]
                    count+=1
    count /=2000        #依照题意取平均值
    return count

count = update(X,Y,0.5)
print(count)