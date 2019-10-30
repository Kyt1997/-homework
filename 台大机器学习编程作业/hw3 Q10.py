# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:22:30 2019

@author: 85402
"""

import math

import numpy as np
u = 0
v = 0

for i in range(5):
    gra_uu = math.e**u + (v**2)*math.e**(u*v) + 2
    gra_vv = 4*math.e**(2*v) + (u**2)*math.e**(u*v) + 4
    gra_uv = math.e**(u*v) + u*v*math.e**(u*v) - 2
    gra_vu = u*v*math.e**(u*v) - 2 + math.e**(u*v)
    
    
    
    gra_u = math.e**u + v*math.e**(u*v) + 2*u - 2*v -3
    gra_v = 2*math.e**(2*v) + u**math.e**(u*v) - 2*u + 4*v - 2
           
    
    hessen = np.array([[gra_uu,gra_uv],[gra_vu,gra_vv]])
    daoshu = np.array([[gra_u],[gra_v]])
    
    u = u - np.matmul(np.linalg.inv(hessen),daoshu)[0][0]
    v = v - np.matmul(np.linalg.inv(hessen),daoshu)[1][0]
    print(u,v)

   
math.e**u + math.e**(2*v) + math.e**(u*v) +u**2 - 2*u*v + 2*v**2 - 3*u -2*v