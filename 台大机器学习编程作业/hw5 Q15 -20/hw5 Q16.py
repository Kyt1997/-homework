# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:11:36 2019

@author: 习近平
"""
from sklearn.svm import SVC
import pandas as pd
import numpy as np

train = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw5 Q15 -20\hw5 train.xlsx',header=None)
test = pd.read_excel(r'C:\Users\85402\Desktop\台大机器学习编程作业\hw5 Q15 -20\hw5 test.xlsx',header=None)
train.columns = [['digit','intensity','symmetry']]
test.columns = [['digit','intensity','symmetry']]

def get_train_data():
    train_y = train['digit']
    train_x = train[['intensity','symmetry']]
    return train_y,train_x

def get_test_data():
    test_y = test['digit']
    test_x = test[['intensity','symmetry']]
    return test_y,test_x

def train_y_transform():
    train_y,train_x = get_train_data()

    save_train_y = []
    for i in range(len(train_x)):
        if train_y.values[i] == 8:
            save_train_y.append(1)
        else:
            save_train_y.append(0)
    train_y = np.array(save_train_y).reshape(7291,)
    
   
    return train_y,train_x

def test_y_transform():
    test_y,test_x = get_test_data()
    save_test_y = []
    for j in range(len(test_x)):
        if test_y.values[j] == 8:
            save_test_y.append(1)
        else:
            save_test_y.append(0)
    test_y = np.array(save_test_y).reshape(len(test_x),)    
    return test_y,test_x




def get_svc_error():
    train_y,train_x = train_y_transform()
    test_y,test_x = test_y_transform()
    clf = SVC(C=0.01,kernel='poly',degree=2,gamma = 1,coef0=1)
    clf.fit(train_x, train_y)
    clf.predict(test_x)
    print(clf.score(test_x,test_y))
     
    return 


def main():
    get_svc_error()
    return
main()

print("0 vs not0 的accuracy是： ",0.8883906327852517)
print("2 vs not2 的accuracy是： ",0.9013452914798207)
print("4 vs not4 的accuracy是： ",0.9003487792725461)
print("6 vs not6 的accuracy是： ",0.9152964623816642)
print("8 vs not8 的accuracy是： ",0.9172894867962132)

"""答案显而易见是选择 8 vs not8 """

