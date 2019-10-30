# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:58:09 2019

@author: 习近平
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:19:47 2019

@author: 习近平
"""

import pandas as pd
import numpy as np
from math import sqrt
from math import log

train = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw6\hw6 adaboost train data.xlsx',header = None)
test = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw6\hw6 adaboost test data.xlsx',header = None)
train.columns = ['x1','x2','y']
test.columns = ['x1','x2','y']


#在训练小gt的时候无需对X,Y进行排序，否则是错误的



def init_U(data,T):
    U_matrix = np.ones((T+1,len(data))) /len(data)
    return U_matrix

def get_epsilon_matrix(T):
    epsilon_matrix = np.zeros((T,1))
    return epsilon_matrix

def get_alpha_matrix(T):
    alpha_matrix = np.zeros((T,1))
    return alpha_matrix

def get_theta_matrix(T):
    theta_matrix = np.zeros((T,1))
    return theta_matrix

def get_s_matrix(T):
    s_matrix = np.zeros((T,1))
    return s_matrix

def get_lowest_Ein_matrix(T):
    lowest_Ein_matrix = np.zeros((T,1))
    return lowest_Ein_matrix

def sort_data(data,feature):
    sort_data = data.sort_values(by = [feature],ascending=True)
    X = sort_data[feature].values.reshape(len(data),1)
    return X
  
    
def get_threshold(X):
    threshold = []
    threshold.append(10000)
    for i in range(len(X)-1):
        threshold.append((X[i][0] + X[i + 1][0])/2)    
    threshold.append(-10000)
    threshold.insert(0,1000)
    return threshold

        
def ada_boost(data,T):
    U_matrix = init_U(data,T)           #初始化权重
    alpha_matrix  = get_alpha_matrix(T)
    theta_matrix = get_theta_matrix(T)
    s_matrix = get_s_matrix(T)
    lowest_Ein_matrix = get_lowest_Ein_matrix(T)
    epsilon_matrix = get_epsilon_matrix(T)
    best_feature_matrix = []
   
    for t in range(T):
        lowest_Ein = 1
        for feature in ['x1','x2']:
            X = sort_data(data,feature)
            threshold = get_threshold(X)                        
            for s in [-1,1]:
                for theta in threshold:
                    Ein = np.dot(U_matrix[t][:].reshape(1,len(data)), s * np.sign(data[feature].values.reshape(len(data),1) - theta)  != data['y'].values.reshape(len(data),1) )
                    if Ein < lowest_Ein:
                        lowest_Ein = Ein
                        best_theta = theta
                        best_s = s
                        best_feature = feature                       
                        best_X = data[feature].values.reshape(len(data),1)
                        best_Y = data['y'].values.reshape(len(data),1)
                        
        print('最低Ein是： ',lowest_Ein,'theta是： ',best_theta,'符号是： ',best_s,'feature是： ',best_feature)
        epsilon_matrix[t][0] = lowest_Ein / sum(U_matrix[t][:])
        theta_matrix[t][0] = best_theta
        alpha_matrix[t][0] = log(sqrt((1-epsilon_matrix[t][0])/epsilon_matrix[t][0]))
        s_matrix[t][0] = best_s
        lowest_Ein_matrix[t][0] = lowest_Ein
        best_feature_matrix.append(best_feature)
        
        for i in range(len(data)):
            if (best_s * np.sign(best_X - best_theta) != best_Y).reshape(len(data[feature].values.reshape(len(data),1)),1)[i][0] == True:
                U_matrix[t+1][i] = U_matrix[t][i] * sqrt((1-epsilon_matrix[t][0])/epsilon_matrix[t][0])
            else:
                U_matrix[t+1][i] = U_matrix[t][i] / sqrt((1-epsilon_matrix[t][0])/epsilon_matrix[t][0])
                
         
    return theta_matrix, alpha_matrix, s_matrix, U_matrix,lowest_Ein_matrix,epsilon_matrix,best_feature_matrix


            
def get_G_Ein(data,T):
    theta_matrix, alpha_matrix, s_matrix, U_matrix, lowest_Ein_matrix ,epsilon_matrix ,best_feature_matrix = ada_boost(data,T)
    
    pre = 0
    for i in range(T):
        
        pre += alpha_matrix[i][0] *(s_matrix[i][0] *np.sign(data[best_feature_matrix[i]]  - theta_matrix[i][0]))
    Ein = sum(np.sign(pre) != data.y)/100
    print('Ein_g1:',lowest_Ein_matrix[0])
    print("经过了",T,"轮迭代后大G的Ein是：",Ein)
    print("U2的summation是： ",sum(U_matrix[1]))
    print("U300的summation是： ",sum(U_matrix[299]))
    print('min epsilon:',np.min(epsilon_matrix))
    
    
    Eout_g1 = sum((s_matrix[0][0]*np.sign(test[best_feature_matrix[0]]  - theta_matrix[0][0]))  != test.y)/1000
    print('g1:',Eout_g1)   
    
    
    
    pre = 0
    for i in range(T):
        pre += alpha_matrix[i][0] * (s_matrix[i][0]*np.sign(test[best_feature_matrix[i]]  - theta_matrix[i][0]))
    E_G = sum(np.sign(pre) != test['y'])/1000
    print('Eout:',E_G)
    
    
get_G_Ein(train,300)




