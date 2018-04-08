# coding=utf-8
'''
python实现简单的大数定理
'''
import random                                  # 导入 random 模块，然后通过 random 静态对象调用该方法
import math                                    # 引用了其他math模块
from collections import Counter
import functools

# 假设我们现在观测一个人掷骰子.这个骰子是公平的，也就是说掷出1~6的概率都是相同的：1/6。他掷了一万次。我们用python来模拟投掷的结果：

# 将每次随机出现的数字放入列表
print('第一步：生成随机数')
times = 1000000
count = []
for i in range(1,times):
    y = random.randint(1,6)
    count.append(y)                    # 将每个 y 放入 count 中

# 统计每个数字出现的次数
print('第二步：统计每个数出现的次数')
list = []
i = 0
for each in count:                    # each 在 count 中没有，则append
    if each not in list:
        list.append(each)
        i += 1
print('List:',list)                           # [6, 5, 2, 3, 1, 4]
a = Counter(count)                    # 计数：Counter({5: 167377, 2: 167009, 6: 166770, 3: 166373, 1: 166271, 4: 166199})
print('计数：',a)

# 将每个数字出现的次数放入列表中
print('第三步：将每个数出现的次数放入列表')
every = []
for b in a:                          # b 即 a 中的每个元素
    every.append(a.get(b))            # a.get(b)  即获得 b 对应的计数值
print('次数：',every)

# 统计每个数字共出现了多少次
print('第四步：计算每个数字出现的概率')
count = functools.reduce(lambda x,y:x+y,every)     # 将 every 中的所有次数加总
if __name__ == '__main__':
 for y in a:
    print("出现%d的概率为" % y,"%.5f%%" % (a.get(y)/count*100))