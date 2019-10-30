# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 22:07:37 2019

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




def get_train_and_val_data(train,which_fold): #include val_x & val_y
    val_x =  train[(which_fold-1) *40 : which_fold*40][['x0','x1','x2']]
    val_x = np.array(val_x).reshape(40,3)
    val_y = train[(which_fold-1) *40 : which_fold*40]['y']
    val_y = np.array(val_y).reshape(40,1)
    
    train_x = np.vstack((train[0:(which_fold-1)*40][['x0','x1','x2']],train[which_fold*40 :][['x0','x1','x2']]))    
    train_x = np.array(train_x).reshape(160,3)
    train_y = np.vstack((train[0:(which_fold-1)*40][['y']],train[which_fold*40 :][['y']]))
    train_y = np.array(train_y).reshape(160,1)
    
    return train_x,train_y,val_x,val_y

def get_cv(train_x,train_y,val_x,val_y,lamda):
    lamda = lamda
    I = np.eye(3,3)
    W_reg = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_x.T,train_x) + lamda*I),train_x.T),train_y.reshape(160,1))
    W_reg = W_reg.reshape(3,1)
    
    pre = np.sign(np.matmul(val_x,W_reg)).reshape(40,1)
    cv = sum(pre != val_y)/40
    return cv


for lam in [10**0,10**-2,10**-4,10**-6,10**-8]:
    total_cv = 0
    for i in range(1,6): 
        train_x,train_y,val_x,val_y = get_train_and_val_data(train,i)
        cv = get_cv(train_x,train_y,val_x,val_y,lamda=lam)
        total_cv += cv
    total_cv /= 5 
    print('lambda是：',lam,'cv是：',total_cv)
    

#19题得到了最佳lambda是10**-8，20题借用Q16-18代码算得0.015和0.02，偷个懒

    
    
    
    


    



