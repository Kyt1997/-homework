# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:54:41 2019

@author: 习近平
"""

import pandas as pd
import numpy as np
import random
d  = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw8\hw8 k-means.xlsx',header = None)

d.columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9']
d['y'] = 0       #用来根据中心点分类





def k_means(k,data,times):
    center_list = []
    for i in range(k):        
        rand = random.randint(0,len(data)-1)  #随机数
        center_x = data[['x1','x2','x3','x4','x5','x6','x7','x8','x9']].values[rand]
        center_y = data.y.values[rand]   #center本身没有y索引
        center_y = i + 1
        center_list.append((center_x,center_y))
        #print(center_list)

    for t in range(times):
        print(t)
        for i in range(len(data)):
            distance_list = []
            for j in range(len(center_list)):
                distance = np.linalg.norm(data[['x1','x2','x3','x4','x5','x6','x7','x8','x9']].values[i] - center_list[j][0])**2
                distance_list.append(distance)
            #print(distance_list)
            data.y.values[i] = center_list[distance_list.index(min(distance_list))][1]
        
        #开始计算各种类的均值
        center_list.clear()
        for i in range(k):
            classes = data[['x1','x2','x3','x4','x5','x6','x7','x8','x9']].values[data.y.values == i+1]
            mean = np.array(np.sum(classes,axis=0)/len(classes)).reshape(1,9)  #均值了
            mean_y = i + 1
            center_list.append((mean,mean_y))
            
    
    err = 0
    for i in range(k):
        classes = data[['x1','x2','x3','x4','x5','x6','x7','x8','x9']][data.y == i+1]
        for l in range(len(classes)):
            err += np.linalg.norm(classes.values[l] - center_list[i][0])**2
    err = err/len(data)
    print(err)
    return err





k_means(10,d,500)
        
    
    
        
            
