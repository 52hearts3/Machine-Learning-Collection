#通过linear_model.Lasso来调用模型
#在Lasso里面我们比较在意两个参数
# 1 alpha  正则化系数
# 2 positive 填布尔值 当这个参数为True时，我们要求Lasso回归出的系数必须是正数，以此来保证我们的alpha一定以增大来控制我们的正则化程度，默认为false
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
for i in [x_train,x_test]:
    i.index=range(i.shape[0])
print(x_train.head())
#先使用线性回归进行拟合
reg=LinearRegression()
reg.fit(x_train,y_train)
print((reg.coef_*100).tolist())
#使用岭回归进行拟合
Ridge_=Ridge(alpha=0)
Ridge_.fit(x_train,y_train)
print((Ridge_.coef_*100).tolist())
#使用Lasso进行拟合
#lasso=Lasso(alpha=0)
#lasso.fit(x_train,y_train)
#print((lasso.coef_*100).tolist())
#使用Lasso会报错
#错误分别是
# 1 正则化系数为0，这样算法不可收敛，如果你想让正则化系数为0，请使用线性回归吧
# 2 没有正则化的坐标下降可能会导致意外结果，不鼓励这样做
# 3 目标函数没有收敛，你也许想要增加迭代次数，使用一个非常小的alpha来拟合模型可能会造成精确度问题

#Lasso对alpha的变化非常敏感
#加大正则项系数，观察模型的系数发生了什么变化
#使用岭回归进行拟合
Ridge_=Ridge(alpha=10**4)
Ridge_.fit(x_train,y_train)
print((Ridge_.coef_*100).tolist())
#使用Lasso进行拟合
lasso=Lasso(alpha=10**4)
lasso.fit(x_train,y_train)
print((lasso.coef_*100).tolist())
#看来，10**4对于lasso来说是一个过于大的取值
lasso=Lasso(alpha=1)
lasso.fit(x_train,y_train)
print((lasso.coef_*100).tolist())
#将系数进行绘图
plt.plot(range(1,9),(reg.coef_*100).tolist(),c='red',label='lr')
plt.plot(range(1,9),(Ridge_.coef_*100).tolist(),c='orange',label='ridge')
plt.plot(range(1,9),(lasso.coef_*100).tolist(),c='k',label='lasso')
plt.xlabel('w')
plt.legend()
plt.show()
#根据图像证明lasso可以将特征压缩为0，而岭回归可以将系数压缩到接近为0，但不会为0
#也就是说，我们可以找到一个适合的alpha，将系数被压缩为0的特征删去