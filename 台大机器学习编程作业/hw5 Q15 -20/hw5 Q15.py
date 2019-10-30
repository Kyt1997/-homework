# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:35:44 2019

@author: 习近平
"""

from sklearn.svm import SVC
import pandas as pd
import numpy as np

train = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw5 Q15 -20\hw5 train.xlsx',header=None)
test = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw5 Q15 -20\hw5 test.xlsx',header=None)
train.columns = [['digit','intensity','symmetry']]
test.columns = [['digit','intensity','symmetry']]

def get_data():
    train_y = train['digit']
    train_x = train[['intensity','symmetry']]
    return train_y,train_x


def label_transform():
    train_y,train_x = get_data()
    y = []
    for i in range(len(train_x)):
        if train_y.values[i] == 0:
            y.append(1)
        else:
            y.append(0)
    train_y = np.array(y).reshape(7291,)
    return train_y,train_x

def get_svc_coef():
    train_y,train_x = label_transform()
    clf = SVC(C=0.01,gamma='auto',kernel='linear')
    clf.fit(train_x, train_y)
    w = clf.coef_               #线性kernel情况下可以直接得出w
    print(np.linalg.norm(w))
    return 

def main():
    get_svc_coef()
    return
main()




    