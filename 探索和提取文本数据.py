from sklearn.datasets import fetch_20newsgroups
data=fetch_20newsgroups()
#fetch_20newsgroups()重要参数
#subset  选择类中包含的数据子集
#输入train表示选择训练集，输入test表示选择测试集，all表示加载所有的数据

#categories  可输入none或者数据所在的目录
#选择一个子集下，不同类型或者不同内容的数据的目录，如果不输入默认none，则会加载全部目录

#shuffle  布尔值，可不填，表示是否打乱样本顺序
#不同类型的新闻,(标签的分类)
print(data.target_names)

import numpy as np
import pandas as pd
categories=['sci.space',#科学技术 太空
            'rec.sport.hockey',#运动 曲棍球
            'talk.politics.guns',#政治 枪支问题
            'talk.politics.mideast']#政治 中东问题
train=fetch_20newsgroups(subset='train',categories=categories)
test=fetch_20newsgroups(subset='test',categories=categories)
print(train.target_names)
#查看有多少篇文章存在
print(len(train.data))
#查看标签
print(np.unique(train.target))
#是否存在样本不均衡问题
for i in np.unique(train.target):
    print(i,(train.target==i).sum()/len(train.target))