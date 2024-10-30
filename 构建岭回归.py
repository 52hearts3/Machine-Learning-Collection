#在linear_model.Ridge来调用

#我们在加州房价数据集上使用线性回归，得出的在训练集上的拟合程度是60%左右
#这个很低的拟合程度是否是由多重共线性造成的？
#如果在岭回归上模型有明显上升趋势，那么是多重共线性造成的

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
print(x_train.head())
reg=Ridge(alpha=1)
reg.fit(x_train,y_train)
score=reg.score(x_test,y_test)
print(score)
#0.6084
#初步判断在加州房价数据集中应该不是共线性问题

#在交叉验证下，与线性回归相比，岭回归结果如何变化
from sklearn.model_selection import cross_val_score
alpha_range=np.arange(1,1001,100)
ridge,lr=[],[]
for alpha in alpha_range:
    reg=Ridge(alpha=alpha)
    linear=LinearRegression()
    regs=cross_val_score(reg,x,y,cv=5,scoring='r2').mean()
    linears=cross_val_score(linear,x,y,cv=5,scoring='r2').mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpha_range,ridge,color='red',label='ridge')
plt.plot(alpha_range,lr,color='blue',label='lr')
plt.title('mean')
plt.legend()
plt.show()
#图像说明加州房价训练集具有非常轻微的多重共线性

#查看模型的方差如何变化
alpha_range=np.arange(1,1001,100)
ridge,lr=[],[]
for alpha in alpha_range:
    reg=Ridge(alpha=alpha)
    linear=LinearRegression()
    regs=cross_val_score(reg,x,y,cv=5,scoring='r2').var()
    linears=cross_val_score(linear,x,y,cv=5,scoring='r2').var()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpha_range,ridge,color='red',label='ridge')
plt.plot(alpha_range,lr,color='blue',label='lr')
plt.title('var')
plt.legend()
plt.show()
#在这个数据集中，随着alpha变大，方差变大，也就是说模型的泛化能力逐渐降低