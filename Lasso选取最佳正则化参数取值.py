#我们使用linear_model.LassoCV来选取alpha参数
#由于Lasso对于alpha的取值更加的敏感，因此我们往往会让alpha在很小的空间中变动，它小到超乎人类的想象
#因此在这个类中有一个重要概念 正则化路径 来设定正则化系数的变动
#在sklearn中，我们通过规定正则化路径的长度（即限制alpha的最小值和最大值之间的比例），以及路径中alpha的个数，让sklearn为我们自动生成alpha的取值

#参数
#eps  正则化路径的长度 默认0.001
#先把eps往小的设，如果效果不行，再慢慢变大
#n_alphas 正则化路径中alpha的个数，默认100
#alphas 需要测试的正则化参数的取值的元组，默认None，当不输入时，会自动使用eps和n_alphas来自动生成带入交叉验证的正则化参数
#cv 交叉验证的次数，默认5折交叉验证

#属性
#alpha_  调出交叉验证选出来的最佳正则化参数
#alphas_  使用正则化路径的长度和路径中alpha的个数来自动生成，用来交叉验证的正则化参数
#mse_path_  返回交叉验证的结果细节
#coef_  调用最佳正则化参数下建立的模型的系数
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge,LinearRegression,Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
house=fetch_california_housing()
x=pd.DataFrame(house.data)
y=house.target
x.columns=house.feature_names
#print(x.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=420)
#恢复索引
for i in [x_train,x_test]:
    i.index=range(i.shape[0])
from sklearn.linear_model import LassoCV
#自己建立lasso的alpha范围
alpha_range=np.logspace(-10,-2,200,base=10)#形成以10为低的指数函数
lasso=LassoCV(alphas=alpha_range,cv=5)
lasso.fit(x_train,y_train)
#查看被选择出来的最佳正则化系数
print(lasso.alpha_)
#调用所有交叉验证的结果
print(lasso.mse_path_.shape)#(200, 5) 返回了每一个alpha取值下的5折交叉验证的结果
print(lasso.mse_path_.mean(axis=1).shape)
#最佳正则化系数下获得的模型的系数结果
print(lasso.coef_)
print(lasso.score(x_test,y_test))

#使用LassoCV自带的正则化路径长度和路径中alpha的个数来自动建立alpha的选择范围
ls_=LassoCV(eps=0.00001,n_alphas=300,cv=5)
ls_.fit(x_train,y_train)
print(ls_.alpha_)
print(ls_.alphas_.shape)#查看所有自动生成的alpha的取值
print(ls_.score(x_test,y_test))
print(ls_.coef_)

#lasso和岭回归都不是为了提升模型的准确度而设计的
#为了提升模型的准确度，我们使用多项式回归