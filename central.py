# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#import pandas as pd
#import seaborn as sns

# 生成任意分布的随机数
print('第一步：生成随机数')
random_data = np.random.randint(1, 7, 10000)      # 生成10000个 1-7 之间的整数，左闭右开[1,7)
mu = random_data.mean()
sigma = random_data.std()
print('  均值：',mu)
print('  方差',sigma)

#随机抽样。先从生成的数据中随机抽取10个数字：
print('第二步：随机抽样10个数')
sample1 = []
for i in range(0, 10):
    sample1.append(random_data[int(np.random.random() * len(random_data))])
print('  样本1：',sample1)                                  # [1, 5, 3, 6, 3, 1, 2, 6, 6, 4]

#现在我们抽取1000组，每组50个。我们把每组的平均值都算出来。
print('第三步：随机抽样100组并画图')

# 50 组
samples = []
samples_mean = []
samples_std = []

for i in range(0, 50):                        # 抽取 50 组
    sample = []
    for j in range(0, 50):                      # 每组50个数
        sample.append(random_data[int(np.random.random() * len(random_data))])
    sample_np = np.array(sample)                # 样本列表

    samples_mean.append(sample_np.mean())
    samples_std.append(sample_np.std())
    samples.append(sample_np)

samples_mean_np0 = np.array(samples_mean)        # 样本均值列表
samples_std_np0 = np.array(samples_std)          # 样本方差列表

# 100组
samples = []
samples_mean = []
samples_std = []

for i in range(0, 500):                        # 抽取 1000 组
    sample = []
    for j in range(0, 50):                      # 每组50个数
        sample.append(random_data[int(np.random.random() * len(random_data))])
    sample_np = np.array(sample)                # 样本列表

    samples_mean.append(sample_np.mean())
    samples_std.append(sample_np.std())
    samples.append(sample_np)

samples_mean_np1 = np.array(samples_mean)        # 样本均值列表
samples_std_np1 = np.array(samples_std)          # 样本方差列表

# 500 组
samples = []
samples_mean = []
samples_std = []

for i in range(0, 1000):                        # 抽取 1000 组
    sample = []
    for j in range(0, 50):                      # 每组50个数
        sample.append(random_data[int(np.random.random() * len(random_data))])
    sample_np = np.array(sample)                # 样本列表

    samples_mean.append(sample_np.mean())
    samples_std.append(sample_np.std())
    samples.append(sample_np)

samples_mean_np2 = np.array(samples_mean)        # 样本均值列表
samples_std_np2 = np.array(samples_std)          # 样本方差列表

# 1000 组
samples = []
samples_mean = []
samples_std = []

for i in range(0, 5000):                        # 抽取 1000 组
    sample = []
    for j in range(0, 50):                      # 每组50个数
        sample.append(random_data[int(np.random.random() * len(random_data))])
    sample_np = np.array(sample)                # 样本列表

    samples_mean.append(sample_np.mean())
    samples_std.append(sample_np.std())
    samples.append(sample_np)

samples_mean_np3 = np.array(samples_mean)        # 样本均值列表
samples_std_np3 = np.array(samples_std)          # 样本方差列表
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.hist(samples_mean_np,bins=10)

fig,(ax0,ax1,ax2,ax3) = plt.subplots(ncols=4,figsize=(16,4))
plt.figure(num=1)
ax0.hist(samples_mean_np0,40,normed=1,histtype='bar',facecolor='green',alpha=0.75)
ax0.set_title('n=50')
ax1.hist(samples_mean_np1,40,normed=1,histtype='bar',facecolor='green',alpha=0.75)
ax1.set_title('n=500')
ax2.hist(samples_mean_np2,40,normed=1,histtype='bar',facecolor='green',alpha=0.75)
ax2.set_title('n=1000')
ax3.hist(samples_mean_np3,40,normed=1,histtype='bar',facecolor='green',alpha=0.75)
ax3.set_title('n=5000')
fig.subplots_adjust(hspace=0.8)              # 调整子图之间的间距

plt.figure(num=2)
mu, sigma = 100, 15
x = mu + sigma*np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

if __name__ == '__main__':
    plt.show()


