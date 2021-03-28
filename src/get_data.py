#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:50:50 2020

@author: Gong Dongsheng
"""

import time
import pandas as pd
import numpy as np
from WindPy import w
w.start()

data_path = "E:/financial_data_science_lession/presentation/data"
stocks_pool = pd.read_csv(data_path+"/stocks_pool.csv", header=None, squeeze=True)

# 别忘了沪深300指数作为market portfolio也要


def get_data(code):
    data = w.wsd(code, "open,high,low,close,volume,vwap,swing,turn,\
                 mkt_freeshares,pe_ttm,pb_mrq,fa_eps_basic,fa_orps,fa_roe_avg,\
                 fa_netprofitmargin_ttm,fa_current,fa_quick,acct_rcv,st_borrow,\
                 tech_MA10,tech_MA20,tech_vema12,tech_vema26,tech_macd,\
                 tech_DIZ,tech_DIF,tech_rsi,tech_bollup,tech_bolldown,tech_bbi,\
                 tech_vosc,tech_cci10,tech_cci20,tech_ATR6,tech_ATR14,tech_obv,\
                 tech_revs10,tech_revs20,tech_bias10,tech_bias20,\
                 holder_sumsqupcttop10,tech_moneyflow20,tech_cry,tech_psy", 
                 "2010-01-01", 
                 "2020-01-01", 
                 "unit=1;rptType=1;Period=M;TradingCalendar=SZSE;PriceAdj=F")
        
    # 确保获取数据的时候无error
    error_code = data.ErrorCode
    if error_code == 0:
        return data
    else:
        return -1


'''
2020年11月14日17时运行记录:
    已经获取了188个数据，从000001.SZ到002136.SZ，剩下的数据超过限制了
'''
count = 0
start = time.time()
count_permin = 0
for code in stocks_pool:
    temp = get_data(code)
    
    # 当存在Error时断掉循环
    if temp == -1:
        print(f"Error occurs at {code}")
        break
    
    df = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=temp.Fields)
    df.to_csv(data_path + f"/series/{code[:-3]}.csv")
    count += 1
    if count % 10 == 0:
        print(count)
        
    # 控制每分钟循环不超过200次，防止达到数据库请求上限
    now = time.time()
    duration = now - start
    if count_permin >= 199 and duration <= 59:
        print("Sleep for a while...")
        time.sleep(61-duration)
        count_permin = 0
        start = time.time()
