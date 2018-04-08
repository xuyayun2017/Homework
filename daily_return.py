# coding=utf-8
'''
   股票收益服从什么分布？
   1. 个股与指数的回归分析（python）
   网址： https://blog.csdn.net/csqazwsxedc/article/details/51336322
          https://www.cnblogs.com/webRobot/p/8471652.html
   2. 用 Python 浅析股票数据
   网址：http://it.dataguru.cn/article-11382-1.html
   3. web从yahoo财经上读取从2001年1月1日开始的德国DAX指数数据；导入了从2014年1月1日开始中信证券的数据
   网址：https://xueqiu.com/3497776981/67554979
   4.正太分布测试/正态性检验
   网址：https://www.cnblogs.com/webRobot/p/6760839.html
   5.上证指数来源：http://quotes.money.163.com/trade/lsjysj_zhishu_000001.html?year=2017&season=1
'''
'''
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
import patsy
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from scipy import stats
import seaborn as sns

import datetime
start = datetime.datetime(2016,1,1)
end = datetime.datetime(2016,12,31)

from pandas_datareader import data,wb
if __name__ == '__main__':
    datass = data.DataReader("000001.SS","yahoo",start,end)
    datajqr = data.DataReader("300024.SZ","yahoo",start,end)
'''
# import xlrd   导excel
# import cav    导入csv
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
if __name__ == '__main__':
        # excel 数据导入 dataframe
        data = pd.read_excel('E:/2018/计量——张涤新/作业三/2017.xlsx')    #
        data.head()
        date = data["日期"]
        close1 = data["前收盘"]
        close2 = data["收盘价"]
        daily_return = data["日收益率"]
        print(close1.describe())             # 打印 count mean std min 25% 50% 75% max
        print(close2.describe())

        # 比较前收盘价与收盘价
        plt.figure(num = 1,figsize = (5,5))
        l1, = plt.plot(date,close1,
                       color = 'blue',
                       label = 'pre_close')
        l2, = plt.plot(date,close2,
                       color = 'red',
                       label = 'close',
                       linestyle='--',)
        plt.legend(handles = [l1,l2],
                   labels = ['pre_close','close'],
                   loc = 'best')
        plt.xlabel("Date")
        plt.ylabel("Close")

        # 日收益率波动图
        plt.figure(num=2, figsize=(5, 5))
        l1, = plt.plot(date, daily_return,
                 color='blue',
                 label='daily return')
        plt.legend(handles =[l1],
                   labels = ['daily return'],
                   loc = 'best')
        plt.xlabel("Date")
        plt.ylabel("Daily return")

        # 日收益率的直方图
        plt.figure(num=3, figsize=(5, 5))
        sns.distplot(daily_return)
        plt.show()







