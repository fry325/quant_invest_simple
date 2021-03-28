#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:16:33 2020

@author: Gong Dongsheng
"""

import os
import matplotlib
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from learn import prepare_data
from lightgbm import LGBMClassifier
os.chdir("E:/financial_data_science_lession/presentation/data")


class Portfolio:
    def __init__(self):
        '''
           amt: 资产总额
          cash: 现金总额
        stocks: 持有股票及其数量
          date: 日期。初始化为2017年1月26日
         price: 储存了股票池所有股票价格的数据表
       suspend: 停牌信息
        '''
        self.amt    = 1000000
        self.cash   = 1000000
        self.stocks = {}
        self.date   = dt.datetime(2017, 1, 26)
        
        self.price  = pd.read_csv("price.csv", index_col=0)
        self.price.index = pd.to_datetime(self.price.index, format='%Y-%m-%d')
        
        self.suspend = {}
        for file_name in os.listdir('suspend'):
            temp = pd.read_csv(f"suspend/{file_name}")
            temp.suspend_date = pd.to_datetime(temp.suspend_date, format='%Y%m%d')
            self.suspend[file_name[:-4]+'.SZ'] = temp
        
    def buy(self, code, num):
        # 买入前判断一下是不是停牌了
        if self.date in list(self.suspend[code].suspend_date):
            print(f"{code} 在 {self.date.year}-{self.date.month}-{self.date.day} 停牌了！")
            return -1
        
        # 买入时分两种情况：是否已经持有该股
        p = self.price[code][self.date]
        if code in self.stocks:
            self.stocks[code] += num
            self.cash -= p * num
        else:
            self.stocks[code] = num
            self.cash -= p * num
        return 1
    
    def sell(self, code, num):
        # 卖出前判断一下是不是停牌了
        if self.date in list(self.suspend[code].suspend_date):
            print(f"{code} 在 {self.date.year}-{self.date.month}-{self.date.day} 停牌了！")
            return -1
        
        # 卖出时
        p = self.price[code][self.date]
        self.stocks[code] -= num
        self.cash += p * num
        return 1
    
    def date_running(self, daysdelta):
        # daysdelta: datetime.timedelta类型
        self.date += daysdelta
        
        stocks_value = 0
        for code in self.stocks:
            p = self.price[code][self.date]
            code_value = p * self.stocks[code]
            stocks_value += code_value
            
        self.amt = self.cash + stocks_value


class invest_score:
    def __init__(self, amts):
        self.amts = amts
        
    def rate_of_return(self):
        # 计算年化收益率
        year = len(self.amts) / 12
        return (self.amts[-1] / self.amts[0]) ** (1 / year) - 1
    
    def sharpe_ratio(self):
        # 计算夏普比率
        ror = self.amts.pct_change().dropna()
        eror = ror.mean()
        stdror = ror.std()
        RF = 3 / 100  # 无风险利率设为3%
        RF_monthly = (1 + RF) ** (1 / 12) - 1
        sr = np.sqrt(12) * (eror - RF_monthly) / stdror
        return sr
    
    def mdd(self):
        # 计算最大回撤
        month = len(self.amts)
        max_dd = 0
        for i in range(1, month):
            amts_i = self.amts[:i]
            high_i = np.max(amts_i)
            now = amts_i[-1]
            dd = (high_i - now) / high_i
            if dd > max_dd:
                max_dd = dd
        return max_dd


if __name__ == "__main__":
    '''
    parameters:
        year: 测试集截止时间（年）
        month: 测试集截止时间（月）
        params: LightGBM模型的参数
        data: 有监督学习数据
        date_series: 储存了回测的所有时间
        portf: 投资组合对象
        amt: 回测的资金，随着时间而变化的序列
    '''
    year = 2017
    month = 1
    interval = [-10, 0, 10]
    params = {
        'boosting_type': 'gbdt', 
        'num_level': 31, 
        'n_estimators': 150, 
        'learning_rate': 0.08, 
        'gamma': 0.05, 
        'max_depth': 6, 
        'sub_sample': 0.9, 
        'random_state': 1
        }
    data = pd.read_csv("data_supervised.csv", index_col=0)
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    date_series = pd.Series(data.date.unique())
    date_series = date_series[date_series>dt.datetime(year=year, month=month, day=15)]
    date_series.index = range(len(date_series))
    portf = Portfolio()
    amts = pd.Series(0, index=date_series)
    amts[0] = portf.amt

    for i in range(len(date_series)):
        # 定义一些循环体内变量
        amt = portf.amt
        stocks = portf.stocks
        year = portf.date.year
        month = portf.date.month
        
        # 拟合模型
        X_train, X_test, y_train, y_test, weight, code = prepare_data(data, year, month, bins=interval)
        model_lgbm = LGBMClassifier(**params)
        model_lgbm.fit(X_train, y_train, weight)
        
        # 找出上涨概率最大的20个股票
        y_proba = model_lgbm.predict_proba(X_test)
        y_proba = pd.DataFrame(data=y_proba, index=code, columns=[0, 1])
        top_20 = y_proba.sort_values(by=[1], ascending=False).index[:20]
        
        # 看看股票池有没有股票，有就先卖出，没有就不管
        if len(stocks) > 0:
            for code in stocks:
                portf.sell(code, stocks[code])
        
        # 买入
        price_top20 = pd.DataFrame(data={'code': top_20})
        temp = []
        for j in range(20):
            temp.append(portf.price[top_20[j]][portf.date])
        price_top20['price'] = temp
        price_top20.sort_values(by='price', ascending=False, inplace=True)
        price_top20.index = range(20)
        num_stocks = 5  # 持有5个股票
        j = 0
        for code in price_top20.code:
            cash_code = portf.cash / (num_stocks - j)
            price_per_hand = price_top20.price[j] * 100
            hand = cash_code // price_per_hand
            buy = portf.buy(code, hand * 100)
            if buy == 1:
                j += 1
            if j >= num_stocks:
                break

        # 时间后移一个月
        print(f"{portf.date.year}年{portf.date.month}月{portf.date.day}日 -> 调仓完成")
        if i != len(date_series) - 1:
            portf.date_running(date_series[i+1]-date_series[i])
            amts[i+1] = portf.amt
        
        
    # 获取深证成指信息
    szcz = pd.read_csv("399001.csv", header=None, squeeze=True)
    szcz.index = amts.index
    amts_g = amts / amts[0]
    szcz_g = szcz / szcz[0]

    # 画图
    matplotlib.rcParams['font.family'] = 'STSong'
    plt.figure(figsize=(30, 20))
    plt.plot(szcz_g, color='r', linewidth=6, label="深证成指")
    plt.plot(amts_g, color='b', linewidth=6, label="量化策略回测")
    plt.fill_between(amts_g.index, 1, amts_g, color='b', alpha=0.2)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=2, alpha=0.5)
    plt.xlabel("时间", fontsize=40)
    plt.ylabel("走势（以1为基准）", fontsize=40)
    plt.tick_params(labelsize=30)
    plt.legend(prop={'size': 40})
    plt.show()


    # 计算年化收益、夏普比率、最大回撤
    print("深证成指")
    szcz_score = invest_score(szcz)
    print("年化收益 -> ", np.round(szcz_score.rate_of_return()*100, 4), "%", sep="")
    print("夏普比率 -> ", np.round(szcz_score.sharpe_ratio(), 4), sep="")
    print("最大回撤 -> ", np.round(szcz_score.mdd()*100, 4), "%", sep="")

    print("\n基于LightGBM算法的量化策略")
    amts_score = invest_score(amts)
    print("年化收益 -> ", np.round(amts_score.rate_of_return()*100, 4), "%", sep="")
    print("夏普比率 -> ", np.round(amts_score.sharpe_ratio(), 4), sep="")
    print("最大回撤 -> ", np.round(amts_score.mdd()*100, 4), "%", sep="")