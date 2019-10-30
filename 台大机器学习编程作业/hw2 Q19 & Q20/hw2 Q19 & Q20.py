# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 18:24:59 2019

@author: 85402
"""



'''本题共有九个维度，一个y，对这九个维度分别实行decision stump,然后看看与y的差别，选出最好的维度的Ein'''


import numpy as np
import pandas as pd
train = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw2 Q19 & Q20\hw2 Q19 train data.xlsx',header=None)
test = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw2 Q19 & Q20\hw2 Q19 test data.xlsx',header=None)

train.columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','y']
test.columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','y']

for z in ['x1','x2','x3','x4','x5','x6','x7','x8','x9']:
    record_s = 0
    temp = 1
    best_theta = 0
    x = list(train[z])
    
        
    theta_list = []
    interval = x.copy()
    interval.insert(0,min(interval)-1)
    interval.append(max(interval) + 1)
    n = 0
    
    while (n <= len(x)):
        theta_list.append(np.random.uniform(interval[n],interval[n+1]))
        n += 1
    
    y = train.y        #x与y绝对不能sorted，这样会打乱原有的对应顺序！！！！！思考了一天才发现这个逻辑漏洞！！
    y = np.array(y)    
    
    
    x = np.array(x)
        
    for s in [1,-1]:
        for theta in theta_list:
            h_x = s * np.sign(x - np.array(theta))
            
            Ein = sum(h_x != y)/len(x)
             
            if Ein < temp:     
                temp = Ein
                best_theta = theta
                record_s = s
    print("Ein最小为：",temp,"theta是：",best_theta,"s是：",record_s)
    
    
#从上面19题的结论可以看出，Ein最小的是x4，为0.25 theta是： 1.7426332549230432 s是： -1，现在将这些东西放进20题目进行计算
for z in ['x1','x2','x3','x4','x5','x6','x7','x8','x9']:
    record_s = 0
    temp = 1
    best_theta = 0
    x = list(test[z])
    
        
    theta_list = []
    interval = x.copy()
    interval.insert(0,min(interval)-1)
    interval.append(max(interval) + 1)
    n = 0
    
    while (n <= len(x)):
        theta_list.append(np.random.uniform(interval[n],interval[n+1]))
        n += 1
    
    y = test.y        #x与y绝对不能sorted，这样会打乱原有的对应顺序！！！！！思考了一天才发现这个逻辑漏洞！！
    y = np.array(y)    
    
    
    x = np.array(x)
        
    
    h_x = s * np.sign(x - np.array(1.7426332549230432))
            
    Ein = sum(h_x != y)/len(x)
            
            
    print("Ein最小为：",Ein)
    
#Ein最小为： 0.362
    