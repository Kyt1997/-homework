# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:14:14 2019

@author: 习近平
"""

from sklearn.svm import SVC
import pandas as pd
import numpy as np
import random
train = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw5 Q15 -20\hw5 train.xlsx',header=None)
test = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw5 Q15 -20\hw5 test.xlsx',header=None)
train.columns = [['digit','intensity','symmetry']]

for g in [1,10,100,1000,10000]:
    E_val = 0
    for i in range(100):
        indices = list(range(len(train)))
        random.shuffle(indices)
        val = train.take(indices[:1000])
        tra = train.take(indices[1000:])
        train_x = tra[['intensity','symmetry']]
        train_y = tra[['digit']]
        val_x = val[['intensity','symmetry']]
        val_y = val[['digit']]
        
        tr_y = []
        v_y = []
        for j in range(len(train_y)):
            if train_y.values[j] == 0 :
                tr_y.append(1)
            else:
                tr_y.append(0)
        tr_y = np.array(tr_y).reshape(len(train_x),)
        
        for k in range(len(val_y)):
            if val_y.values[k] == 0:
                v_y.append(1)
            else:
                v_y.append(0)
        v_y = np.array(v_y).reshape(len(val_x),)
        
        clf = SVC(C=0.1,kernel='rbf',degree=2,gamma = g,coef0=0)
        clf.fit(train_x, tr_y)
        clf.predict(val_x)    
        E_val +=  1 - clf.score(val_x,v_y)
    print("gamma = ",g,"的情况下E_val = ", E_val/100)
    


