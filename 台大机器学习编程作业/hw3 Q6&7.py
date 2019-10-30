# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 12:00:19 2019

@author: 85402
"""
from math import e
u = 0
v = 0
n = 0.01
for i in range(5):
    print("这是第",i,"次实验")
    gra_u = e**u + (e**(u*v)) * v + 2*u -3 + 2*v
    gra_v = (e**(2*v)) * 2 + (e**(u*v)) * u - 2*u + 4*v -2
    u = u - n*gra_u
    v = v - n*gra_v   
    
    print(gra_u,gra_v)
    
    
E = e**u + e**(2*v) + e**(u*v) + u**2 - 2*u*v + 2*v**2 - 3*u - 2*v
print(E)
