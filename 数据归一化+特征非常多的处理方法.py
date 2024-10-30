#当数据x按照最小值中心化后，再按极差（最大值-最小值）缩放，数据移动了最小值个单位，并且会被收敛到0到1之间
#这个过程，就叫数据归一化（Normalization）    正则化是regularization
import numpy as np
from sklearn.preprocessing import MinMaxScaler
data=[[-1,2],[-0.5,6],[0,10],[1,18]]
import pandas as pd
print(pd.DataFrame(data))
#实现归一化
scaler=MinMaxScaler()
scaler.fit(data)
result=scaler.transform(data)#通过接口导出结果
print(result)
s=scaler.inverse_transform(result)#将归一化结果逆转
print(s)
#使用feature_range(5,10)将数据归一化到5到10的范围
scaler=MinMaxScaler(feature_range=(5,10))
result=scaler.fit_transform(data)
print(result)
#当x中的特征数量非常多的时候，fit会报错并表示数据量太大了计算不了
#此时用partial_fit为接口

#使用numpy进行归一化
x=np.array([[-1,2],[-0.5,6],[0,10],[1,18]])
print(x.min())#返回所有数据中的最小值
print(x.min(axis=0))#返回两列的最小值  axis=0是跨列计算
print(x.min(axis=1))#返回每一行的最小值  axis=1是跨行运算
#归一化
x_nor=(x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
print(x_nor)
#逆转
x_return=x_nor*(x.max(axis=0)-x.min(axis=0))+x.min(axis=0)
print(x_return)