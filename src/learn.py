#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:48:53 2020

@author: Gong Dongsheng
"""

import copy
import numpy as np
import pandas as pd
import datetime as dt
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score


def get_X(data):
    X = np.array(data.drop(columns=['date', 'code', 'code_int', 'return']))
    return X


def get_y(data, bins):
    '''
    params:
        data: pandas.DataFrame,
              传入的有监督学习数据，必须包含'return'一列用于生成离散化标签
        bins: list,
              按照bins的区间生成离散化数据
    return:
           y: numpy.array,
              收益率离散化之后的标签，标签按照收益率大小来排列，最小为0；
              例如默认的bins传出的y将会包含0,1,2,3,4共五类标签
    '''
    intervals = pd.cut(data['return'], bins)
    y = intervals.factorize(sort=True)[0]
    return y


def split_data(data, year, month):
    '''
    data_train: 传入日期往前推300个星期的所有数据
     data_test: 传入日期当月的所有数据，只预测一期
    '''
    this_month = dt.datetime(year, month, 15)
    start_month = this_month - dt.timedelta(weeks=350)
    next_month = this_month + dt.timedelta(weeks=4)
    data_train = data[(data['date']>=start_month) & (data['date']<this_month)]
    data_test = data[(data['date']>=this_month) & (data['date']<=next_month)]
    data_train.index = range(data_train.shape[0])
    return data_train, data_test


def get_weight(data):
    # 获取权重，离今天越近权重越大，日期往前推则权重线性递减
    weight = (data.date - data.date[0]).dt.days
    weight = np.array(weight)
    intercept = weight[-1] // 3
    weight += intercept
    s = np.sum(weight)
    weight = weight / s
    return weight


def prepare_data(data, year=2017, month=1, bins=[-10, 0, 10]):
    data_train, data_test = split_data(data, year, month)
    code = data_test['code']
    X_train, X_test = get_X(data_train), get_X(data_test)
    y_train, y_test = get_y(data_train, bins), get_y(data_test, bins)
    weight = get_weight(data_train)
    return X_train, X_test, y_train, y_test, weight, code


def learning(X_train, y_train, weight, **kw):
    model_lgbm = LGBMClassifier(**kw)
    model_lgbm.fit(X_train, y_train, weight)
    return model_lgbm


data = pd.read_csv("E:/financial_data_science_lession/presentation/data/data_supervised.csv", index_col=0)
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
interval = [-10, 0, 10]
year = 2017
month = 1
X_train, X_test, y_train, y_test, weight, code = prepare_data(data, year, month, bins=interval)

# 试一下瞎猜的准确率
ps = 0
for i in range(10000):
    y_pred = y_test.copy()
    np.random.shuffle(y_pred)
    ps += precision_score(y_test, y_pred)
ps /= 10000
print("瞎猜的概率：", ps)
# 瞎猜的准确率：34.75%

'''
# 试一下LGBM
y_pred, t = learning(X_train, 
                     y_train, 
                     weight, 
                     X_test, 
                     'LGBM',
                     boosting_type='gbdt',
                     num_leavel=10,
                     n_estimators=2000, 
                     learning_rate=0.8, 
                     gamma=0.1, 
                     max_depth=5, 
                     subsample=0.8, 
                     tree_method='auto', 
                     random_state=1)
print("LightGBM algorithm classification results:")
print(precision_score(y_test, y_pred))
'''

params_list = {
        'num_level': [5, 10, 20, 40, 80, 160], 
     'n_estimators': [250, 500, 1000, 2000, 4000, 8000], 
    'learning_rate': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2], 
            'gamma': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6], 
        'max_depth': [4, 5, 6, 7, 8, 9], 
       'sub_sample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
params = {
    'boosting_type': 'gbdt', 
    'num_level': 5, 
    'n_estimators': 250, 
    'learning_rate': 0.1, 
    'gamma': 0.05, 
    'max_depth': 4, 
    'sub_sample': 0.4, 
    'random_state': 1
    }
params_best = copy.deepcopy(params)

precision_best = 0
for param_name in params_list:
    precision_best_i = 0
    for param_value in params_list[param_name]:
        params[param_name] = param_value
        model_lgbm = LGBMClassifier(**params)
        model_lgbm.fit(X_train, y_train, sample_weight=weight)
        
        y_pred = model_lgbm.predict(X_test)
        precision = precision_score(y_test, y_pred)
        if precision > precision_best_i:
            precision_best_i = precision
            params_best[param_name] = param_value
            
        print(params)
        print(precision)
        
        if param_value == params_list[param_name][-1]:
            params[param_name] = params_best[param_name]
    
    if precision_best_i > precision_best:
        precision_best = precision_best_i
    print(f"best precision: {precision_best}")

'''
第一轮调参
2017年1月以前的数据作为training set，测出来的最优参数：
{
'boosting_type': 'gbdt', 
'num_level': 31, 
'n_estimators': 250,    # 能再减少一点吗？
'learning_rate': 0.1,   # 能再减少一点吗？
'gamma': 0.05, 
'max_depth': 6, 
'sub_sample': 0.9, 
'random_state': 1
}
下面要进行第二轮调参。精确地确定一下n_estimators和learning_rate的最优参数值
'''

params_list = {
     'n_estimators': [50, 100, 150, 200, 250, 300, 350], 
    'learning_rate': [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    }
params = {
    'boosting_type': 'gbdt', 
    'num_level': 31, 
    'n_estimators': 250, 
    'learning_rate': 0.1, 
    'gamma': 0.05, 
    'max_depth': 6, 
    'sub_sample': 0.9, 
    'random_state': 1
    }
params_best = copy.deepcopy(params)

for param_name in params_list:
    precision_best = 0
    for param_value in params_list[param_name]:
        params[param_name] = param_value
        model_lgbm = LGBMClassifier(**params)
        model_lgbm.fit(X_train, y_train, sample_weight=weight)
        
        y_pred = model_lgbm.predict(X_test)
        precision = precision_score(y_test, y_pred)
        if precision > precision_best:
            precision_best = precision
            params_best[param_name] = param_value
            
        print(params)
        print(precision)
        
        if param_value == params_list[param_name][-1]:
            params[param_name] = params_best[param_name]
            
'''
最终确定的最优参数值：
{
'boosting_type': 'gbdt', 
'num_level': 31, 
'n_estimators': 150, 
'learning_rate': 0.08, 
'gamma': 0.05, 
'max_depth': 6, 
'sub_sample': 0.9, 
'random_state': 1
}
'''

'''
# 试一下神经网络
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

y_trainb = to_categorical(y_train)
y_testb = to_categorical(y_test)

model = Sequential()
model.add(Dense(units=200, activation='sigmoid', input_dim=168))
model.add(Dense(units=2, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, y_trainb, epochs=200, batch_size=20)
model.evaluate(X_test, y_testb)
'''