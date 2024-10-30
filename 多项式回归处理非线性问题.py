import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
#创建测试数据集
line=np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)

#原始特征矩阵的拟合结果
linear_r=LinearRegression()
linear_r.fit(x,y)
print(linear_r.score(x,y))
print(linear_r.score(line,np.sin(line)))

#进行多项式拟合
d=5
poly=PolynomialFeatures(degree=d)
x_=poly.fit_transform(x)
line_=poly.transform(line)
#训练数据的拟合
linear_r_=LinearRegression()
linear_r_.fit(x_,y)
print(linear_r_.score(x_,y))
#测试数据的拟合
print(linear_r_.score(line_,np.sin(line)))

#画图
import matplotlib.pyplot as plt
fig,ax1=plt.subplots(1)
ax1.plot(line,linear_r.predict(line),linewidth=2,color='green',label='linear regression')
ax1.plot(line,linear_r_.predict(line_),linewidth=2,color='red',label='polynomial regression')
#将原数据上的拟合绘制在图像上
ax1.plot(x[:,0],y,'o',c='k')
ax1.legend(loc='best')
ax1.set_ylabel('regression output')
ax1.set_xlabel('input feature')
ax1.set_title('linear regression vs poly')
plt.tight_layout()
plt.show()

#多项式的degree我们通常取5，6，7，8，9 不超过10
#如何选择画交叉验证的学习曲线