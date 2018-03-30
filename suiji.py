# coding=utf-8
#假设我们现在观测一个人掷骰子.这个骰子是公平的，也就是说掷出1~6的概率都是相同的：1/6。他掷了一万次。我们用python来模拟投掷的结果：
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns

random_data = np.random.randint(1, 7, 10000)
print(random_data.mean()) # 打印平均值
print(random_data.std())  # 打印标准差

#随机抽样。先从生成的数据中随机抽取10个数字：
sample1 = []
for i in range(0, 10):
    sample1.append(random_data[int(np.random.random() * len(random_data))])
print(sample1) # 打印出来

#现在我们抽取1000组，每组50个。我们把每组的平均值都算出来。
samples = []
samples_mean = []
samples_std = []

for i in range(0, 1000):
    sample = []
    for j in range(0, 50):
        sample.append(random_data[int(np.random.random() * len(random_data))])
    sample_np = np.array(sample)
    samples_mean.append(sample_np.mean())
    samples_std.append(sample_np.std())
    samples.append(sample_np)

samples_mean_np = np.array(samples_mean)
samples_std_np = np.array(samples_std)

print(samples_mean_np)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.hist(samples_mean_np,bins=10)
#plt.title('')

plt.show()
