# coding=utf-8
'''
    Python实现逻辑回归(Logistic Regression in Python)
    网址：http://www.powerxing.com/logistic-regression-in-python/
    R语言以及Stata方法：（idre） https://stats.idre.ucla.edu/
'''
'''
    逻辑回归是一项可用于预测二分类结果(binary outcome)的统计技术，广泛应用于金融、医学、犯罪学和其他社会科学中。
    逻辑回归使用简单且非常有效，你可以在许多机器学习、应用统计的书中的前几章中找到个关于逻辑回归的介绍。逻辑回归在许多统计课程中都会用到。
    需要的包：
    pandas: 直接处理和操作数据的主要package
    statsmodels: 统计和计量经济学的package，包含了用于参数评估和统计测试的实用工具
    pylab: 用于生成统计图
    numpy: Python的语言扩展，定义了数字的数组和矩阵
'''
'''
    实例：辨别不同的因素对研究生录取的影响。
    数据集中的前三列可作为预测变量(predictor variables)：gpa，gre,rank(母校声望)
    第四列admit则是二分类目标变量(binary target variable)，它表明考生最终是否被录用。
'''
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
if __name__ == '__main__':
    # 加载数据：1.使用网址pd.read_csv("http://cdn.powerxing.com/files/lr-binary.csv")；2.pd.read_csv(./Data/binary.csv)
    print('1.加载数据')
    df = pd.read_csv("./Data/binary.csv")
    # 浏览数据集
    print(df.head())
    # 重命名 rank 列，因为 Dataframe中有个方法名也叫 rank
    df.columns = ['admit','gre','gpa','prestige']
    print("列名：",df.columns)

    print('2.统计摘要&查看数据')
    # 可以使用pandas的函数describe来给出数据的摘要
    # crosstab可方便的实现多维频率表(frequency tables)(有点像R语言中的table)。你可以用它来查看不同数据所占的比例。
    print('统计性描述：')
    print(df.describe())
    print('频率表，表示prestige与admit的值相应的数量关系')
    print(pd.crosstab(df['admit'],df['prestige'],rownames=['admit']))

    #  plot all of the columns
    df.hist()
    pl.show()

    print('3.虚拟变量')
    # 虚拟变量，也叫哑变量，可用来表示分类变量、非数量因素可能产生的影响。
    # pandas提供了一系列分类变量的控制。我们可以用get_dummies来将”prestige”一列虚拟化。get_dummies为每个指定的列创建了新的带二分类预测变量的DataFrame。
    # 本例中，prestige有四个级别：1，2，3以及4（1代表最有声望），prestige作为分类变量更加合适。
    # 当调用get_dummies时，会产生四列的dataframe，每一列表示四个级别中的一个。
    print('将 Prestige 设为虚拟变量：')
    dummy_ranks = pd.get_dummies(df['prestige'],prefix='prestige') # 将prestige设为虚拟变量：prestige_1  prestige_2  prestige_3  prestige_4
    print(dummy_ranks.head())

    # 为逻辑回归创建所需的 Dataframe
    # 除admit、gre、gpa外，加入了上面常见的虚拟变量（注意，引入的虚拟变量列数应为虚拟变量总列数减1，减去的1列作为基准）
    print('4.为逻辑回归创建所需的 Dataframe:')
    cols_to_keep = ['admit','gre','gpa']
    data = df[cols_to_keep].join(dummy_ranks.ix[:,'prestige_2':]) # admit  gre   gpa  prestige_2  prestige_3  prestige_4
    print(data.head())
    # 需要自行添加逻辑回归所需要的常数 intercept 变量。statsmodels 实现的逻辑回归需要显示指定。
    data['intercept'] = 1.0

if __name__ == '__main__':
    print('5.执行逻辑回归')
    # 首先指定要预测变量的列，接着指定模型用于做预测的列，剩下的就由算法包去完成了。
    # 本例中要预测的是admit列，使用到gre、gpa和虚拟变量prestige_2、prestige_3、prestige_4。prestige_1作为基准，所以排除掉，以防止多元共线性(multicollinearity)和引入分类变量的所有虚拟变量值所导致的陷阱(dummy variable trap)。
    # 指定作为训练变量的列，不含目标列
    train_cols = data.columns[1:]             # Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)
    logit = sm.Logit(data['admit'],data[train_cols])
    # 拟合模拟
    result = logit.fit()
    print('Logit 回归结果:')
    print(result.summary())
    # 结果包含两部分：
    # 上半部分给出了和模型整体相关的信息，包括因变量的名称（Dep. Variable: admit）、模型名称（Model: Logit）、拟合方法（Method: MLE 最大似然估计）等信息；
    # 下半部分则给出了和每一个系数相关的信息，包括系数的估计值（coef）、标准误（std err）、z统计量的值、显著水平（P>|z|）和95%置信区间。
    print('查看每个系数的置信区间:')
    print(result.conf_int())

    print("Probit 回归结果:")
    probit = sm.Probit(data['admit'],data[train_cols])
    result = probit.fit()
    print(result.summary())
    # 使用每个变量系数的指数来生成odds ratio，可知变量每单位的增加、减少对录取几率的影响。
    # 例如，如果学校的声望为2，则我们可以期待被录取的几率减少大概50%。
    print('6.相对危险度（odds ratio)')
    print(np.exp(result.params))

    # 也可以使用置信区间来计算系数的影响，来更好地估计一个变量影响录取率的不确定性。
    print('7.使用置信区间计算系数的影响')
    params = result.params
    conf = result.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%','97.5%','OR']
    print(np.exp(conf))
