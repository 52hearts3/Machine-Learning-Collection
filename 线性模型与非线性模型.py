#作为线性模型的代表，我们可以从线性回归的方程中总结出模型的特点，其自变量都是一次项

#线性回归在非线性数据集的表现如何
#我们观察线性回归和决策树在非线性数据集上的表现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#创建需要拟合的数据集
rnd=np.random.RandomState(42)
x=rnd.uniform(-3,3,size=100)#random.uniform,从输入的任意两个整数范围中取出size个随机数
y=np.sin(x)+rnd.normal(size=len(x))/3 #random.normal  生成size个服从正态分布的随机数
#+rnd.normal(size=len(x))的目的是添加噪音
#使用散点图观察数据的样子
plt.scatter(x,y,marker='o',c='k',s=20)
plt.show()

#为后续建模做准备，上课Learn只接受二维以上的数组作为特征矩阵输入
x=x.reshape(-1,1)
#使用原始数据进行建模
linear=LinearRegression()
linear.fit(x,y)
tree=DecisionTreeRegressor(random_state=0)
tree.fit(x,y)
#放置画布
fig,ax1=plt.subplots(1)
#创建测试数据集
line=np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)
#将测试数据集带入predict接口进行预测
ax1.plot(line,linear.predict(line),linewidth=2,color='green',label='linear regression')
ax1.plot(line,tree.predict(line),linewidth=2,color='red',label='decision tree')
#将原数据上的拟合绘制到画布上
ax1.plot(x[:,0],y,'o',c='k')
ax1.legend(loc='best')
ax1.set_ylabel('regression output')
ax1.set_xlabel('input feature')
ax1.set_title('result before discretization')
plt.tight_layout()
plt.show()
#看来在非线性数据集上，决策树的效果比线性回归好
#事实上，机器学习远远比我们想象的灵活得多
#线性模型可以用来拟合非线性数据，而非线性模型也可以用来拟合线性数据
#更神奇的是，有的算法没有模型也可以处理各类数据，
# 而有的模型既可以是线性模型，又可以是非线性模型（支持向量机）