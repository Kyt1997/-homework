# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 12:59:01 2018

@author: 85402
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 17:17:08 2018

@author: 85402
"""
import random
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


def verify_on_train(train_data_x,train_data_y,W,b):
                #本函数将每一次迭代后的W，b用在test_set上面做检验
    error = 0
    for j in range(500):
        out_in_train = np.matmul(W,train_data_x.values[j].reshape(4,1))+b
        if train_data_y.values[j]*out_in_train <=0:
            error += 1
    error_rate_train = error/500
    return error_rate_train

    
def verify_on_test(test_data_x,test_data_y,W,b):
    error = 0
    for o in range(500):
        out_in_test = np.matmul(W,test_data_x.values[o].reshape(4,1))+b
        if test_data_y.values[o]*out_in_test <=0:
            error += 1
    error_rate_test = error/500
    return error_rate_test


def naive_pla(train_data_x,train_data_y,test_data_x,test_data_y):
    all_error_test = 0
    for z in range(2000):       
        W,b = init()  
        count = 0
        current_error = 1
        sample = np.random.randint(len(train_data_x),size=len(train_data_x))
        while(count<100):
            for i in sample:
                outcome = np.matmul(W,train_data_x.values[i].reshape(4,1)) +b
                if outcome *train_data_y.values[i] <=0:
                    W = W+train_data_x.values[i] * train_data_y.values[i]
                    b = b+train_data_y.values[i]            
                    #完成了一次更新，接下来该把这两个参数拿去train_set上面
                    error_rate_train = verify_on_train(train_data_x,train_data_y,W,b)
                    #单组W，b的错误率
                    if error_rate_train < current_error:            
                        #记录错误
                        current_error = error_rate_train
                        pocket_w = W
                        pocket_b = b
                    count += 1
                    print("当前是第",z,"次实验的第",count,"次迭代")
                
        error_rate_test = verify_on_test(test_data_x,test_data_y,pocket_w,pocket_b)
        print("当前测试机错误率是：",error_rate_test)        
        all_error_test += error_rate_test
    all_error_test /= 2000
    return all_error_test

a = naive_pla(train_data_x,train_data_y,test_data_x,test_data_y)
print(a)