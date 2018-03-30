# coding=utf-8
'''
python实现简单的大数定理
'''
import random
import math
from collections import Counter
import functools

times = 10000000
count = []#将每次随机出现的数字放入列表
for i in range(1,times):
    y = random.randint(1,6)
    count.append(y)
list = []#统计每个数字出现的次数
i = 0
for each in count:
    if each not in list:
        list.append(each)
        i += 1
a = Counter(count)
every = []#将每个数字出现的次数放入列表中
for b in a:
    every.append(a.get(b))
count = functools.reduce(lambda x,y:x+y,every)#统计每个数字共出现了多少次
for y in a:
    print("出现%d的概率为" % y,"%.5f%%" % (a.get(y)/count*100))