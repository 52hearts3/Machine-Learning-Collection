#多项式回归模拟支持向量机对数据进行升维

#通过preprocessing.PolynomiaFeatures调用多项式变化，这是线性模型中的升维工具
#参数
#degree  多项式中的次数。默认为2
#interaction_only  填布尔值，是否只产生交互项，默认为false
#include_bias  填布尔值，是否产出与截距项相乘的x0，默认为True

from sklearn.preprocessing import PolynomialFeatures
import numpy as np
#如果原始数据是一维的
x=np.arange(1,4).reshape(-1,1)
#二次多项式，参数degree控制多项式的次方
poly=PolynomialFeatures(degree=2)
x_=poly.fit_transform(x)
print(x_)#变成了三行三列
#由此推断，假设多项式转化的次数是n，数据将会被转化成形如 1,x,x^2,x^3 ......x^n的形式
#拟合出的方程也可以被写为y=w0*x0+w1*x+w2*x^2+w3*x^3+......+wn*x^n,x0=1

#参数include_bias就是控制x0生成的
#三次多项式，不带与截距项相乘的x0
poly_3=PolynomialFeatures(degree=3,include_bias=False)
x_3=poly_3.fit_transform(x)
print(x_3)#少了最前面的一列

#为什么我们会希望不生成与截距相乘的x0呢
#对于多项式回归来说，我们已经为线性回归准备好了x0，但是线性回归并不知道
xxx=PolynomialFeatures(degree=3).fit_transform(x)
print(xxx.shape)
#设定y
rnd=np.random.RandomState(42)
y=rnd.randn(3)
print(y)
#使用线性模型拟合
from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(xxx,y)
print(linear.coef_)
print(linear.intercept_)
#因为include_bias=True的情况下，第一列为1，x0应该等于截距项才对
#但线性回归并没有把多项式生成的x0当作是截距项
#所以我们选择关闭多项式回归中的include_bias
#也可以选择关闭线性回归中的fit_intercept

#尝试关闭后的
linear=LinearRegression(fit_intercept=False)
linear.fit(xxx,y)
print(linear.coef_)
print(linear.intercept_)#截距为0

#尝试二维数据
x=np.arange(6).reshape(3,2)
poly=PolynomialFeatures(degree=2)
x_=poly.fit_transform(x)
print(x_)
#结果为y=w0+w1*x1+w2*x2+w3*x1*x2+w4*x1^2+w5*x^2

#想要总结规律，尝试三次多项式
poly=PolynomialFeatures(degree=3)
x_=poly.fit_transform(x)
print(x_)
#不难发现，当我们进行多项式转换的时候，多项式会产出到最高次数为止的所有最低次项
#注意 x1*x2是二次项

#使用interaction_only  填布尔值，是否只产生交互项 会在一定程度上避免共线性
poly=PolynomialFeatures(degree=2,interaction_only=True)
x_=poly.fit_transform(x)
print(x_)#没有了x1^2,x2^2

#随着特征数量和degree增加，特征的数量会呈指数级增长
