# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:57:53 2019

@author: 习近平
"""
from mxnet import autograd,init
from mxnet.gluon import nn,Trainer
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet import nd
import pandas as pd
train  = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw8\hw8 nnet train.xlsx',header = None)
test = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw8\hw8 nnet test.xlsx',header = None)
train.columns = ['x1','x2','y']

test.columns = ['x1','x2','y']


train_x = nd.array(train[['x1','x2']].values)
train_y = nd.array(train.y.values)

test_x = nd.array(test[['x1','x2']].values)
test_y = nd.array(test.y.values)



batch_size =1


trainset = gdata.ArrayDataset(train_x,train_y)

train_iter = gdata.DataLoader(trainset,batch_size, shuffle=True)

testset = gdata.ArrayDataset(test_x,test_y)
test_iter = gdata.DataLoader(testset,500,shuffle= False)


def get_Eout(T,M,lr):
    
    
    net = nn.Sequential()
    net.add(nn.Dense(M,activation='tanh'))
    net.add(nn.Dense(1,activation='tanh'))
    net.initialize(init.Normal(sigma=0.01))
    trainer = Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
    loss = gloss.L2Loss()

    
    for t in range(T):
        if t % 1000 == 0:
            print('t:',t)
        for X,y in train_iter:        
            with autograd.record():
                l = loss(net(X),y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(X),y)
    print('训练完成,开始检验')
        
        
    for tX,ty in test_iter:
        l = loss(net(tX),ty).mean().asscalar()
        print('Eout;',l)
       
    return l,M

c = []
for m in [1,6,11,16,21]:
    l,M = get_Eout(T=50000,M=m,lr=0.1)
    c.append((l,M))
    
mi = min(c,key=lambda x:x[0])
print(mi)



