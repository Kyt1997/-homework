# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:26:07 2019

@author: 习近平
"""



import pandas as pd
import numpy as np
import random
train  = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw7\hw7 train data.xlsx',header = None)
test = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw7\hw7 test data.xlsx',header = None)
train.columns = ['x1','x2','y']
test.columns = ['x1','x2','y']




class Node:
    def __init__(self,theta,feature,value = None):
        self.theta = theta
        self.feature = feature
        self.value = value
        self.left_node = None
        self.right_node = None
     
        
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


def gini(Y):
    length = Y.shape[0]
    if length == 0:
        return 0
    else:
        pos = len(Y[Y ==1])
        neg = len(Y[Y == -1])
        return 1 - (pos/length)**2 - (neg/length)**2


def cut_D(D,feature,theta):
    D1 = D[D[feature] <= theta]
    D2 = D[D[feature] > theta]
    return D1,D2

def decision_stump(D):
    b_branch = 10000
    for feature in ['x1','x2']:
        X = sort_data(D,feature)
        threshold = get_threshold(X)        
        for theta in threshold:
            D1,D2 = cut_D(D,feature,theta)
    
            branch = len(D1)/len(D) * gini(D1.y.values) + len(D2)/len(D) * gini(D2.y.values)
            #print('feature = ',feature,',theta = ',theta)
            if branch < b_branch:
                b_branch = branch
                b_feature = feature
                b_theta = theta

                
    return b_feature,b_theta
                


def dtree(l,d,first):
     
    if len(l) == 0:
            return first
    
    c_node = l.pop(0)
    
    if c_node.value == None and c_node.theta == None and c_node.feature == None:
        best_feature,best_theta = decision_stump(d[0])
        c_node = Node(best_theta,best_feature)
        #print(best_feature,best_theta)
        D1,D2 = cut_D(d[0], best_feature , best_theta)
        d.append(D1)
        d.append(D2)
        d.pop(0)
        first = c_node
        
       
    best_feature,best_theta = decision_stump(d[0]) 
    
    if np.sum(d[0].y.values != 1) ==0 or np.sum(d[0].y.values != -1) == 0:
        node = Node(None,None,value=d[0].y.values[0])
        c_node.left_node = node
        d.pop(0)
    else:
        node = Node(best_theta,best_feature)
        c_node.left_node = node
        l.append(c_node.left_node)
        D1,D2 = cut_D(d[0], best_feature , best_theta)
        d.append(D1)
        d.append(D2)
        d.pop(0)        
                     
    
    best_feature,best_theta = decision_stump(d[0]) 
    
    if np.sum(d[0].y.values!= 1) ==0 or np.sum(d[0].y.values != -1) == 0:           #不应该是branch
        node = Node(None,None,value=d[0].y.values[0])
        c_node.right_node = node
        d.pop(0)
    else:
        node = Node(best_theta,best_feature)
        c_node.right_node = node
        l.append(c_node.right_node)
        D1,D2 = cut_D(d[0], best_feature , best_theta)
        d.append(D1)
        d.append(D2)
        d.pop(0)       
        
                
    return dtree(l,d,first)        


def validation(nd,data,i):
    #print('validating.....wait')

    current_node = nd.pop(0)
    #print(current_node.feature,current_node.theta,current_node.value)
    
    if current_node.feature == None and current_node.theta == None and current_node.value != None:    
        #print(data.values[i])
        #print("第",i,"个值：",current_node.value)
        return current_node.value
    
    
    if data[current_node.feature].values[i] <= current_node.theta:
        nd.append(current_node.left_node)
        return validation(nd,data,i)
    
    if data[current_node.feature].values[i] > current_node.theta:
        nd.append(current_node.right_node)
        return validation(nd,data,i)

  
  
def Ein_Eout(D,val_x):
    first_node = Node(None,None)
    l = [first_node]
    d = [D]
    first = 1
    head_node = dtree(l,d,first)
    #print(head_node.feature,head_node.theta)
    
   
    pre = []
    
    for i in range(len(val_x)):
        nd = [head_node]
        predict = validation(nd,val_x,i)
        #print("第",i,"个预测：",predict)
        pre.append(predict)
    
    pre = np.array(pre).reshape(len(pre),1)
    single_err = sum(pre != val_x.y.values.reshape(len(pre),1))/len(pre)
    
    
 
    return pre,single_err

#Ein_Eout(train,train)        #Ein = 0.0
#Ein_Eout(train,test)     #Eout = 0.126


def bagging(D):
    indices = []
    for i in range(len(D)):
        sample = random.randint(0,len(D)-1)
        indices.append(sample)
    
    
    
    N_prime = train.iloc[indices,:]
    
    
    return N_prime


def RF(D,val_x):
    avg_Egt = 0
    total_pre = 0
    for exp in range(1,2):
        print('这是第',exp,'次实验')
        for t in range(1,301):
            print('这是第',t,'颗树')
            N_prime = bagging(D)
            pre,single_err= Ein_Eout(N_prime,val_x)
            avg_Egt += single_err
            total_pre += pre
    #avg_Egt /= 30000       
    total_pre = sign(total_pre)
    print(total_pre)
    Ein_RF = sum(total_pre != val_x.y.values.reshape(len(val_x),1))/len(val_x)
    Ein_RF /= 1
    
    #print("平均的Ein(gt)： ",avg_Egt)
    
    print('Ein(RF)是：', Ein_RF)
    
    return
RF(train,train)    #Ein(gt)=0.05466666666666666 ,Ein(RF)=0
RF(train,test)      #Eout(RF)=0.076