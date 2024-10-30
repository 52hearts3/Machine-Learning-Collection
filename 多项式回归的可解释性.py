#我们可以调用重要接口get_feature_names_out()
#来调用生成的新特征矩阵上各个特征的名称，以便我们解释模型
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
x=np.arange(9).reshape(3,3)
poly=PolynomialFeatures(degree=5)
poly.fit(x)
#print(poly.get_feature_names_out())

#在加州房价数据集上看看效果
from sklearn.datasets import fetch_california_housing
import pandas as pd
house=fetch_california_housing()
x=pd.DataFrame(house.data)
y=house.target
x.columns=house.feature_names
poly=PolynomialFeatures(degree=4)
poly.fit(x,y)
print(poly.get_feature_names_out())#默认把x替换为列名
x_=poly.transform(x)
reg=LinearRegression()
reg.fit(x_,y)
coef=reg.coef_
print(coef)
print([*zip(poly.get_feature_names_out(),reg.coef_)])
#放到dataframe中进行排序
coeff=pd.DataFrame([poly.get_feature_names_out(),reg.coef_.tolist()]).T
coeff.columns=['feature','coef']
#按照coef进行排序
coeff.sort_values(by='coef',inplace=True)
print(coeff.head())

#我们可以发现，不仅数据的可解释性还存在，我们还可以通过这样的手段做特征工程--创造特征
#多项式回归帮助我们创造了一系列特征之间的相乘的组合，若能够找出组合起来后对标签贡献巨大的特征，那我们就是创造了新的有效特征

print(reg.score(x_,y))
#在其他模型上查看效果‘
from sklearn.ensemble import RandomForestRegressor
rfc=RandomForestRegressor()
rfc.fit(x,y)
print(rfc.score(x,y))#0.9745485641850516

#那么为什么还要学习线性模型
#因为运行速度快

#在现实中，多项式变化在疯狂增加数据维度的同时，也增加了过拟合的可能性，因此多项式变化多与能够处理过拟合的线性模型如岭回归，lasso等连用