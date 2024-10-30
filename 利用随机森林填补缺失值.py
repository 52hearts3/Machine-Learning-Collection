#在sklearn中sklearn.impute.Simplelmputer来将一些常用的数据填入数据中
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer  #用于填补缺失值
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
fetch_california_housing=fetch_california_housing()
data=fetch_california_housing.data
target=fetch_california_housing.target
feature_names=fetch_california_housing.feature_names
df=pd.DataFrame(data=data,columns=feature_names)
print(df)
x_full,y_full=data,target
n_samples=x_full.shape[0]
n_features=x_full.shape[1]
#填入缺失值
rng=np.random.RandomState(0)
missing_rate=0.5
n_missing_samples=int(np.floor(n_samples*n_features*missing_rate))
print(n_missing_samples)
#考虑到乘后可能有小数，用np.floor向下取整，返回.0格式的浮点数

#所有数据随机遍布在各行各列中，而一个缺失的数据会需要一个行索引和一个列索引
#如果能够创造一个数组，包含82560个分布在0到20640中间的行索引和82560个分布在0到8之间的列索引
#然后我们用0，均值和随机森林来填补这些缺失值，然后查看回归结果如何
missing_features=rng.randint(0,n_features,n_missing_samples)
missing_samples=rng.randint(0,n_samples,n_missing_samples)
x_missing=x_full.copy()
y_missing=y_full.copy()
#把空值加到数据集中
x_missing[missing_samples,missing_features]=np.nan
x_missing=pd.DataFrame(x_missing)
print(x_missing)

#使用均值填补缺失值!!!
imp_mean=SimpleImputer(missing_values=np.nan,strategy='mean')#输入median中位数  输入most_frequent众数
x_missing_mean=imp_mean.fit_transform(x_missing)#fit_transform返回填完均值的框架(变为array格式)
df2=pd.DataFrame(x_missing_mean)
print(df2.isnull().sum())#布尔值false=0，true=1 如果有空值，那么最后结果一定比0大

#使用0进行填补!!!
imp_0=SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0)
x_missing_0=imp_0.fit_transform(x_missing)
df3=pd.DataFrame(x_missing_0)
print(df3.isnull().sum())

#使用随机森林回归填补缺失值!!!
#对于一个有n个特征的数据来说，其中特征T有缺失值，我们就把特征T当作标签，其他n-1个特征和原本的标签组成新的特征矩阵，对于T来说
#它没有缺失的部分，就是我们的y_train,这部分数据既有标签也有对应的特征，而它缺失的部分只有特征没有标签，就是我们预测的部分
#特征T不缺失的值对应的其他n-1个特征+本来的标签 : x_train
#特征T不缺失的值: y_train
#特征T缺失的值对应的其他n-1个特征+本来的标签: x_test
#特征T缺失的值  :  y_test
#这种做法对某一个特征大量缺失，其他特征缺很完整的情况非常适合

#如果数据中除了特征T之外，其他特征也有缺失值怎么办
#答案是遍历所有的特征，从缺失最少的开始进行填补（因为填补缺失最少的特征所需要的准确信息最少）
# 。填补一个特征时，先将其他特征的缺失值用0代替，每完成一次回归预测，就将预测值放到原本的特征矩阵中，再继续填补下一个特征。
# 每一次填补完毕，有缺失值的特征会减少一个，所以每次循环后，需要用0来填补的特征就越来越少。
# 当进行到最后一个特征时（这个特征应该是所有特征中缺失值最多的），已经没有任何的其他特征需要用0来进行填补了，而我们已经使用回归为其他特征填补了大量有效信息，
# 可以用来填补缺失最多的特征。
x_missing_reg=x_missing.copy()
#找出数据集中缺失值从小到大排列的顺序
sort_index = np.argsort(x_missing_reg.isnull().sum()).values#返回每一列的空值 后面是列，前面是空值个数
#argsort 返回从小到大排序的顺序对应的索引
print(sort_index)
for i in sort_index:
    #构建新特征矩阵（没有被选去填充的特征+原始的标签）和新标签（被选去填充的特征）
    df=x_missing_reg
    #新标签
    fillc=df.iloc[:,i]
    #新特征矩阵
    df=pd.concat([df.iloc[:,df.columns!=i],pd.DataFrame(y_full)],axis=1)#y_full指原来没有缺失值的标签
    #在新特征矩阵中，对含有缺失值的列，用0进行填补
    df_0=SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0).fit_transform(df)#df_0为array格式
    #找出训练集和测试集
    #y_train是被选中要填充的特征中（现在是我们的标签），存在的非空值
    y_train=fillc[fillc.notnull()]
    #y_test是被选中的要填充的特征中（现在是我们的标签），存在的空值
    #我们需要的不是y_test的值，而是y_test所带的索引
    y_test=fillc[fillc.isnull()]
    #x_train是在新特征矩阵上，被选出来的要填充的特征的非空值对应的的记录
    x_train=df_0[y_train.index,:] #df_0为矩阵
    #x_test是在新特征矩阵上，被选出来的要填充的特征的空值对应的的记录
    x_test=df_0[y_test.index,:]
    #用随机森林回归来填补缺失值
    rfc=RandomForestRegressor()
    rfc.fit(x_train,y_train)
    #得到我们填补空值的值
    y_predict=rfc.predict(x_test)
    #将补好的特征返回到我们原始的特征矩阵中
    x_missing_reg.loc[x_missing_reg.iloc[:,i].isnull(),i]=y_predict
print(pd.DataFrame(x_missing_reg))

#对几种填充方式进行打分
x=[x_full,x_missing_mean,x_missing_0,x_missing_reg]
mse=[]
for i in x:
    estimator=RandomForestRegressor()
    scores=cross_val_score(estimator,i,y_full,scoring='neg_mean_squared_error',cv=10).mean()
    mse.append(scores*-1)
print(mse)