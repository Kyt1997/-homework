# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:41:42 2019

@author: 85402
"""


import numpy as np
error_list = []         #用于储存5000次实验的5000个最小E_in
best_theta_list = []          #用于储存最小E_in的theta
for exp in range(5000):
    temp = 1            #用于储存单次实验中比之前更小的E_in
    
    
    
    print(exp,"次实验")
    x = sorted(np.random.uniform(-1,1,20))      
    
    
    
    interval = x.copy()
    interval.append(1)
    interval.insert(0,-1)
    
    
    
    n = 0
    theta_list = []
    while(n <= 20):
        theta_list.append(np.random.uniform(interval[n],interval[n+1]))
        n += 1
    
        
    f = np.sign(x)
    y = []
    for i in range(len(f)):
        a = np.random.uniform(0,1)
        if a <= 0.2:
           y.append(f[i] * (-1))
        else:
            y.append(f[i])
    
    y = np.array(y)

    for s in [1,-1]:
        for theta in theta_list:
            h_x = s * np.sign(np.array(x) - np.array(theta))
            error = sum(h_x != y)/20
            if error < temp:
                temp = error
                best_theta = theta
    best_theta_list.append(best_theta)      
    error_list.append(temp)     #存进去本次实验最小的E_in
print(sum(error_list)/5000)

'''下面是18题'''

save_Eount = []

temp_for_Eout = 1
    
for S in [1,-1]:
    for t in best_theta_list:
        Eout = 0.5 + 0.3*S*(np.abs(t) - 1)
        if Eout < temp_for_Eout:
            temp_for_Eout = Eout
save_Eount.append(temp_for_Eout)

print(save_Eount)
        
    

            
        
        
