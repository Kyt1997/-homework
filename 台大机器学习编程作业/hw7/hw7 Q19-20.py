# -*- coding: utf-8 -*-
"""
Created on Sun May 12 09:12:37 2019

@author: 习近平
"""

import pandas as pd
import numpy as np
import random
train  = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw7\hw7 train data.xlsx',header = None)
test = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw7\hw7 test data.xlsx',header = None)
train.columns = ['x1','x2','y']
test.columns = ['x1','x2','y']

def sign(v):
    for i in range(len(v)):
        if v[i][0] <= 0 :
            v[i][0] = -1
        else:            
            v[i][0] = 1
    return v

        
                     
def sort_data(D,feature):
    sort_data = D.sort_values(by = [feature],ascending=True)
    X = sort_data[feature].values.reshape(len(D),1)
    return X

def get_threshold(X):
    threshold = [-1000]
    
    for i in range(len(X)-1):
        threshold.append((X[i][0] + X[i + 1][0])/2)    
    threshold.append(10000)
    return threshold


def decision_stump(D):
    b_Ein = 10000
    for feature in ['x1','x2']:
        X = sort_data(D,feature)
        for s in [-1,1]:
            threshold = get_threshold(X)        
            for theta in threshold:
                       
                branch = s * sign(D[feature].values.reshape(len(D),1) - theta)
                Ein = sum(branch != D.y.values.reshape(len(D),1))/len(D)
                #print('feature = ',feature,',theta = ',theta)
                if Ein <= b_Ein:
                    b_Ein = Ein
                    b_feature = feature
                    b_theta = theta   
                    b_s = s
    #print('EIN is:',b_Ein)
    return b_feature,b_theta,b_s



def bagging(D):
    indices = []
    for i in range(len(D)):
        sample = random.randint(0,len(D)-1)
        indices.append(sample)
    
    
    
    N_prime = train.iloc[indices,:]
    
    
    return N_prime


def get_err(D,val):
    all_err = 0
    for exp in range(10):
        print("第",exp,"次实验")
        total_pre = 0
        for t in range(300):
            print("第",t,"棵树")
            
            N_prime = bagging(D)
            b_feature,b_theta,b_s= decision_stump(N_prime)
            
            #print(b_feature,b_theta)
            
            
            pre = b_s * (val[b_feature].values.reshape(len(val),1) - b_theta)
            pre = pre.reshape(len(pre),1)
            pre = sign(pre)
            
            
            #single_err = sum(pre != val.y.values.reshape(len(val),1))/len(val)
            #print(single_err)
            total_pre += pre
         
        total_err = sum(sign(total_pre) != val.y.values.reshape(len(val),1))/len(val)
        all_err += total_err
        
    avg_err = all_err/10
    print(avg_err)
    
    
        
    return

get_err(train,train)
get_err(train,test)

