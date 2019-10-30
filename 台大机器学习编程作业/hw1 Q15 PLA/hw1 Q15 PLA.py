# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 16:32:46 2018

@author: 85402
"""

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


def update(train_x,train_y):
    W,b = init()
    count = 0
    for j in range(10):         #多遍历几个epoch，这样可以完全将所有的误分类点纠正过来
        for i in range(len(X)):
            pre_y = np.matmul(W,X.values[i].reshape(4,1)) + b
            if pre_y * Y.values[i] <=0:
                W = W+Y.values[i]*X.values[i]
                b = b+Y.values[i]
                count+=1
    return count

count = update(X,Y)
print("update次数为：",count)


       
        
        