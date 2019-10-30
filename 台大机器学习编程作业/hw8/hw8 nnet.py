# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:07:58 2019

@author: 习近平
"""

import pandas as pd
import numpy as np
import math

train  = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw8\hw8 nnet train.xlsx',header = None)
test = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw8\hw8 nnet test.xlsx',header = None)
train.columns = ['x1','x2','y']

test.columns = ['x1','x2','y']


train_x = train[['x1','x2']].values.reshape(2,len(train))
train_y = train.y.values.reshape(1,len(train))
train_y.shape[1]

test_x = test[['x1','x2']].values.reshape(2,len(test))
test_y = test.y.values.reshape(1,len(test))
'''T = 50000
M = 1
learning_rate = 0.1
r = 0.1'''


a = np.zeros((3,1))
b = np.ones((3,1))
c = sum((a-b)**2)
print(c)



def init_W(M,x,r):
    W1 = np.random.uniform(-r,r,size=(x.shape[0],M))
    b1 = np.random.uniform(-r,r,size=(M,1))
    W2 = np.random.uniform(-r,r,size=(M,1))
    b2 = np.random.uniform(-r,r,size=(1,1))
    return W1,b1,W2,b2

def score(W,x,b):
    return np.matmul(W.T,x) + b
    
def tanh(s):
    return (math.e**(s) - math.e**(-s))/(math.e**(s) + math.e**(-s))   
   
def backward(x,y,T,learning_rate,M,r,experiment):
    W1,b1,W2,b2 = init_W(M,x,r)
    print('init_W1:',W1.shape)
    print('init_b1:',b1.shape)
    print('init_W2:',W2.shape)
    print('init_b2:',b2.shape)
    for t in range(T):
        #print('第',t,'轮')
        s1 = score(W1,x,b1)
        #print("s1: ",s1.shape)
        x1 = tanh(s1)       #中间层
        #print("x1:",x1.shape)
        s2 = score(W2,x1,b2)
        #print("s2:",s2.shape)
        output = tanh(s2)
        #print("output: ",output.shape)
        
        dw1 = np.matmul((2*(output -y)*(1-output**2)*x1 * (1-x1**2)), x.T).reshape(M,2) * (1/25)
        
        #print("dw1 :",dw1.shape)
       
        db1 = np.sum(2*(output -y)*(1-output**2)* x1*(1-x1**2),axis=1).reshape(M,1) * (1/25)
        
        #print("db1: ",db1.shape)




        
        dw2 = np.sum(2*(output -y)*(1-output**2)*x1,axis=1).reshape(M,1) * (1/25)
        #print("dw2: ",dw2.shape)
        
        db2 = 2*(output -y)*(1-output**2)
        db2 = np.sum(db2,axis=1).reshape(1,1) * (1/25)
        #print("db2: ",db2.shape)
        
        W1 = W1 - learning_rate * dw1.T
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dw2
        b2 = b2 - learning_rate * db2
        
        '''print('W1:',W1.shape)
        print('b1:',b1.shape)
        print('W2:',W2.shape)
        print('b2:',b2.shape)
        print("s1: ",s1.shape)
        print("x1:",x1.shape)
        print("s2:",s2.shape)
        print("output: ",output.shape)
        print("dw1 :",dw1.shape)
        print("db1: ",db1.shape)
        print("dw2: ",dw2.shape)
        print("db2: ",db2.shape)'''

    return W1,b1,W2,b2


def get_E(val_x,x,val_y,T,learning_rate,M,r,experiment):
    b_W1,b_b1,b_W2,b_b2 = backward(train_x,train_y,T,learning_rate,M,r,experiment)

    '''print("b_W1:",b_W1.shape)
    print("b_W2",b_b1.shape)'''
    
    
    s1 = score(b_W1,val_x,b_b1)
    #print(s1.shape)
    x1 = tanh(s1)
    #print(x1.shape)
    s2 = score(b_W2,x1,b_b2)
    predict = tanh(s2)
   

    valy = val_y.reshape(1,val_y.shape[1])
    #print(valy.shape)
    predict = predict.reshape(1,valy.shape[1])
    #print(predict)
    err = np.sum((predict - valy)**2) * (1/val_x.shape[1])
    print("在",M,"的情况下的err是： ",err)
    return err
        
for m in [1,6,11,16,21]:
    #print("now m= ",m)
    get_E(val_x = test_x , x =train_x , val_y = test_y , T=50000, learning_rate=0.1 , M=m , r=0.1 ,experiment=50)    
    
 


   