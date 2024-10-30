#我们用岭计图来判断正则化参数的最佳取值
#岭计图是一个以正则化参数为横坐标，线性模型求解的系数w为纵坐标的图像
#岭计图认为，线条交叉越多，说明特征之间的多重共线性越高，我们应该选择系数较为平稳的喇叭口所对应的alpha取值作为最佳正则化参数的取值
#但是非常不推荐使用岭计图

#所以我们使用交叉验证来选择最佳正则化系数
#我们就选择交叉验证下均方误差最小的正则化系数alpha
#我们使用linear_model.RidgeCV来对岭回归进行交叉验证
#重要参数
#alphas  需要测试的正则化参数取值的元组
#scoring  用来进行交叉验证的模型评估指标，默认是R**2，可自行调整
#store_cv_results 是否保存每次交叉验证的结果，默认为false
#cv  交叉验证的模式，默认为None，表示默认进行留一交叉验证（最适合岭回归）
#可以输入Kfold对象和StratifiedFold对象来进行交叉验证
#注意，仅仅为None时，每次交叉验证的结果才能被保存下来
#当cv有值存在（不是none）时，store_cv_values无法被设定为True

#重要属性
#alpha_  查看交叉验证选中的alpha
#cv_results_  调用所有交叉验证的结果，只有当store_cv_results=True时才能够调用，因此返回的结构上(n_samples,n_alphas)
#score  调用Ridge类不进行交叉验证的情况下返回的R**2

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge,LinearRegression,Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import RidgeCV
house=fetch_california_housing()
x=pd.DataFrame(house.data)
y=house.target
x.columns=house.feature_names
Ridge_=RidgeCV(alphas=np.arange(1,1001,100),store_cv_results=True)
Ridge_.fit(x,y)
#输出无关交叉验证的岭回归结果
print(Ridge_.score(x,y))
#调用所有交叉验证的结果
print(Ridge_.cv_results_.shape)#(20640, 10) 10列指的是在10种不同的alpha下的结果
#进行平均后查看每个正则化系数取值下的交叉验证的结果
print(Ridge_.cv_results_.mean(axis=0))
#查看最佳正则化系数
print(Ridge_.alpha_)