# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:06:46 2019

@author: Administrator
"""

import numpy as np
a = 0
for num in range(1000):
    print("第",num,"次试验")
    
    N = 1000
    
    
    x1 = []
    x2 = []
    for i in range(N):                      #生成x
        x1.append(np.random.uniform(-1,1))
        x2.append(np.random.uniform(-1,1))
    
    x1 = np.array(x1).reshape(1000,1)           #堆叠成1000*3的向量
    x2 = np.array(x2).reshape(1000,1)
    x0 = np.ones((1000,1))
    X = np.hstack((x1,x2,x0))   
    
    
    target_data= []
    for i in range(N):
        target_data.append(np.sign(X[i][0]**2 + X[i][1]**2 - X[i][2]*0.6))       #target_function
    
    target_data = np.array(target_data).reshape(1000,1)
    
    
    noise_data = []
    for i in range(N):                                  #按10%添加噪声
        flip = np.random.random()
        if flip <= 0.1:
            noise_data.append(target_data[i] * -1)
        else:
            noise_data.append(target_data[i])
    
           
#至此，下面是预测部分    
    
    W_lin = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),noise_data)
    
    
    pre = np.sign(np.matmul(X,W_lin))
    pre = np.array(pre)
    
    a += sum(sum(pre != noise_data)/1000)
    
print(a/1000)



#以下是14题
g1 = []
g2 = []
g3 = []
g4 = []
g5 = []
for i in range(1000):
    g1.append(np.sign(-1 -1.5*X[i][0] + 0.08*X[i][1] + 0.13*X[i][0]*X[i][1] + 0.05*X[i][0]*X[i][0] + 0.05*X[i][1]*X[i][1]))
    g2.append(np.sign(-1 -0.05*X[i][0] + 0.08*X[i][1] + 0.13*X[i][0]*X[i][1] + 1.5*X[i][0]*X[i][0] + 1.5*X[i][1]*X[i][1]))
    g3.append(np.sign(-1 -0.05*X[i][0] + 0.08*X[i][1] + 0.13*X[i][0]*X[i][1] + 1.5*X[i][0]*X[i][0] + 15*X[i][1]*X[i][1]))
    g4.append(np.sign(-1 -0.05*X[i][0] + 0.08*X[i][1] + 0.13*X[i][0]*X[i][1] + 15*X[i][0]*X[i][0] + 1.5*X[i][1]*X[i][1]))
    g5.append(np.sign(-1 -1.5*X[i][0] + 0.08*X[i][1] + 0.13*X[i][0]*X[i][1] + 0.05*X[i][0]*X[i][0] + 1.5*X[i][1]*X[i][1]))





g1 = np.array(g1).reshape(1000,1)
g2 = np.array(g2).reshape(1000,1)
g3 = np.array(g3).reshape(1000,1)
g4 = np.array(g4).reshape(1000,1)
g5 = np.array(g5).reshape(1000,1)
    
print("g1: ",sum(g1 != noise_data))
print("g2: ",sum(g2 != noise_data))  
print("g3: ",sum(g3 != noise_data))  
print("g4: ",sum(g4 != noise_data))  
print("g5: ",sum(g5 != noise_data))      
    

#由此知道g2是最优的,np.sign(-1 -0.05*X[i][0] + 0.08*X[i][1] + 0.13*X[i][0]*X[i][1] + 1.5*X[i][0]*X[i][0] + 1.5*X[i][1]*X[i][1])
total = 0
for num in range(1000):
    print("第",num,"次试验")
    
    N = 1000
    
    
    x1 = []
    x2 = []
    for i in range(N):                      #生成x
        x1.append(np.random.uniform(-1,1))
        x2.append(np.random.uniform(-1,1))
    
    x1 = np.array(x1).reshape(1000,1)           #堆叠成1000*3的向量
    x2 = np.array(x2).reshape(1000,1)
    x0 = np.ones((1000,1))
    X = np.hstack((x1,x2,x0))   
    
    
    target_data= []
    for i in range(N):
        target_data.append(np.sign(X[i][0]**2 + X[i][1]**2 - X[i][2]*0.6))       #target_function
    
    target_data = np.array(target_data).reshape(1000,1)
    
    
    test_noise_data = []
    for i in range(N):                                  #按10%添加噪声
        flip = np.random.random()
        if flip <= 0.1:
            test_noise_data.append(target_data[i] * -1)
        else:
            test_noise_data.append(target_data[i])
    
    test_noise_data = np.array(test_noise_data)        
    pre_on_test = []
    for i in range(1000):
        pre_on_test.append(np.sign(-1 -0.05*X[i][0] + 0.08*X[i][1] + 0.13*X[i][0]*X[i][1] + 1.5*X[i][0]*X[i][0] + 1.5*X[i][1]*X[i][1]))
    pre_on_test = np.array(pre_on_test).reshape(1000,1)
    
    total += sum(pre_on_test != test_noise_data)/1000
print(total/1000)
    
    