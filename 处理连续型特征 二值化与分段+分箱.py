#根据阈值将数据二值化（将特征值设置为0或1），用于处理连续型变量，大于阈值的映射为1，小于阈值的映射为0
#默认阈值为0时，特征中所有正值都映射为1
#二值化是对文本计数数据的常见方法
import pandas as pd
data=pd.read_csv(r'D:\game\sklearn\数据预处理\Narrativedata.csv',index_col=0)
print(data)
from sklearn.preprocessing import Binarizer
from sklearn.impute import SimpleImputer
x=data.iloc[:,0].values.reshape(-1,1)#iloc[:,0]取出的是series，是一列索引一列值，用values把值取出来 再升维
g=SimpleImputer(strategy='mean')
x=g.fit_transform(x)#填补缺失值
tran=Binarizer(threshold=38).fit_transform(x)
print(tran)
data.iloc[:,0]=tran
print(data.head())

#分箱处理KBinsDiscretizer
#1  n_bins  每个特征中分箱的个数，默认为5，一次会被运用到所有导入的特征
#2  encode 编码方式  默认onehot
#onehot 做哑变量，之后返回一个稀疏数组，每一列是一个特征中的一个类别
#ordinal  每个特征的每个箱都被编码为整数，返回的每一列是一个特征，每个特征下含有不同的整数编码的箱的矩阵
#onehot_dense 做哑变量，之后返回一个密集数组
#3  strategy定义箱宽的方式
#默认quantile
#uniform 表示等宽分箱，即每个特征中的每个箱的最大值之间的差为（特征.max()-特征.min()）/(n_bins)
#quantile  表示等位分箱 即每个特征中的每个箱内的样本数量都相同
#kmeans  表示按聚类分箱 每个箱中的值到最近的一维k均值聚类的簇心的距离都相同

from sklearn.preprocessing import KBinsDiscretizer
est=KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='quantile')
s=est.fit_transform(x)
print(s)#变为0，1，2三个箱子
print(set(s.ravel()))#set可以查看有多少不重复的元素
est=KBinsDiscretizer(n_bins=3,encode='onehot',strategy='uniform')
s2=est.fit_transform(x).toarray()
print(s2)#生成了三列矩阵，因为分为了三箱