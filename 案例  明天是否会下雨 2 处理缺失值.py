import pandas as pd
import numpy as np
x_train=pd.read_csv(r'D:\game\sklearn\支持向量机  下\x_train.csv',index_col=0)
x_test=pd.read_csv(r'D:\game\sklearn\支持向量机  下\x_test.csv',index_col=0)
y_train=pd.read_csv(r'D:\game\sklearn\支持向量机  下\y_train.csv',index_col=0)
y_test=pd.read_csv(r'D:\game\sklearn\支持向量机  下\y_test.csv',index_col=0)
print(x_train.head())
print(y_train.head())

#处理缺失值
#传统的，如果是分类型特征，我们一般选择众数填补，如果是连续型特征，我们一般选择用均值填补
#我们一般使用训练集上的众数和均值对训练集和测试集同时填补
#查看缺失值情况
print(x_train.isnull().mean())
#首先找出，我们的分类型特征有哪些
#分类型特征一般类型为object
cate=x_train.columns[x_train.dtypes=='object'].tolist()
print(cate)
#除了特征类型为object的特征们，还有虽然用数字表示，但本质为分类型特征的云层遮蔽程度
cloud=['Cloud9am','Cloud3pm']
cate=cate+cloud
print(cate)

#对于分类型特征，我们用众数来填补
from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
si.fit(x_train.loc[:,cate])
#使用训练集中的众数来同时填补训练集和测试集
x_train.loc[:,cate]=si.transform(x_train.loc[:,cate])
x_test.loc[:,cate]=si.transform(x_test.loc[:,cate])
#查看分类型特征是否依然存在缺失值
print(x_train.loc[:,cate].isnull().mean())#填补成功

#处理分类型变量，将分类型变量编码
#将所有的分类型变量编码为数字，一个类别是一个数字
from sklearn.preprocessing import OrdinalEncoder#只允许二维以上的数据进行输入
oe=OrdinalEncoder()
oe.fit(x_train.loc[:,cate])
#用训练集的编码结果来编码训练集和测试集
#在这里如果测试集特征矩阵报错，就说明测试集中出现了训练集中从未有过的类别
x_train.loc[:,cate]=oe.transform(x_train.loc[:,cate])
x_test.loc[:,cate]=oe.transform(x_test.loc[:,cate])
print(x_train.head())

#处理连续型变量  填补缺失值
#在现实中，我们填补连续型变量很少用算法
#因为算法填补解释性不强，别人可能会不懂，算法填补的时间也慢
#在比赛中可以尝试用算法填补
#首先找到分类型变量
col=x_train.columns.tolist()
for i in cate:
    col.remove(i)
print(col)
from sklearn.impute import SimpleImputer
imp_mean=SimpleImputer(missing_values=np.nan,strategy='mean')
imp_mean.fit(x_train.loc[:,col])
#分别用训练集的训练结果对训练集和测试集填补
x_train.loc[:,col]=imp_mean.transform(x_train.loc[:,col])
x_test.loc[:,col]=imp_mean.transform(x_test.loc[:,col])
print(x_train.isnull().mean())

#处理连续型变量  无量纲化
#一定不要再分类型变量上进行
col.remove('Month')
from sklearn.preprocessing import StandardScaler
#标准化不改变数据的分布，不会吧数据变为正态分布
ss=StandardScaler()
ss.fit(x_train.loc[:,col])
x_train.loc[:,col]=ss.transform(x_train.loc[:,col])
x_test.loc[:,col]=ss.transform(x_test.loc[:,col])
print(x_train.head())
x_train.to_csv(r'D:\game\sklearn\支持向量机  下\x_train_final.csv')
x_test.to_csv(r'D:\game\sklearn\支持向量机  下\x_test_final.csv')