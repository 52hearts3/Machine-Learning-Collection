#当数据x按照均值中心化，再按照标准差缩放，数据就会服从为均值为0，方差为1的正态分布（标准正态分布） 这个过程叫标准化
from sklearn.preprocessing import StandardScaler
data=[[-1,2],[-0.5,6],[0,10],[1,18]]
scaler=StandardScaler()
scaler.fit(data)
#查看均值的属性mean_
print(scaler.mean_)#按列计算
#查看方差的属性var_
print(scaler.var_)#按列计算
x_std=scaler.transform(data)
print(x_std)
print(x_std.mean())
print(x_std.std())
#逆转
s=scaler.inverse_transform(x_std)
print(s)

#标准化和归一化选哪个
#大多数机器学习中选标准化，因为中心化对异常值非常敏感
#在pca 聚类，逻辑回归（分类），支持向量机  上 二分类  最强大的机器学习算法，神经网络这些算法中，标准化往往是最好的选择
#归一化在不涉及距离度量，梯度，协方差计算以及数据需要被压缩到特定区间时使用广泛

#此外，还有其他处理方法
#如果只希望压缩数据，不需要中心化，用MaxAbsScaler
#在异常值多，噪声非常大时，我们可能会选择用分位数来无量纲化RobustScaler