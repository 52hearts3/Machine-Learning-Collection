#在没有其他算法或者预处理的情况下，线性模型在非线性数据集上表现得非常糟糕
#我们可以通过分箱是线性模型处理非线性数据

#让线性回归在非线性数据集上提升表现的核心方法之一是对数据进行分箱，也就是离散化
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
x=x.reshape(-1,1)

from sklearn.preprocessing import KBinsDiscretizer
#将数据进行分箱
enc=KBinsDiscretizer(n_bins=10,encode='onehot')
#encode='onehot'  使用做哑变量的方式做离散化
#之后会返回一个稀疏矩阵(m行，n_bins列)，每一列是一个特征中的一个类别，含有该类别的样本表示为1，不含的表示为0
x_binned=enc.fit_transform(x)
#print(x_binned)
#使用pandas打开稀疏矩阵
import pandas as pd
df=pd.DataFrame(x_binned.toarray())
print(df.head())
#使用原始数据进行建模
linear=LinearRegression()
linear.fit(x,y)
tree=DecisionTreeRegressor(random_state=0)
tree.fit(x,y)
#创建测试数据集
line=np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)
#我们需要对创建分箱后的测试集，按照已经建好的分箱模型将line进行分箱
line_binned=enc.transform(line)
#将两张图象绘制在一起，布置画布
fig,(ax1,ax2)=plt.subplots(ncols=2,sharey=True,figsize=(10,4))#sharey=True 让两张图共享y轴上的刻度
#在图一中布置在原始数据上的建模结果
ax1.plot(line,linear.predict(line),linewidth=2,color='green',label='linear regression')#linewidth=2 线宽为2
ax1.plot(line,tree.predict(line),linewidth=2,color='red',label='decision tree')
ax1.plot(x[:,0],y,'o',c='k')
ax1.legend(loc='best')
ax1.set_ylabel('regression output')
ax1.set_xlabel('input feature')
ax1.set_title('result before discretization')
#使用分箱数据进行建模
linear_=LinearRegression()
linear_.fit(x_binned,y)
tree_=DecisionTreeRegressor(random_state=0)
tree_.fit(x_binned,y)
#进行预测，在图二上布置分箱数据在进行预测的结果
ax2.plot(line,linear_.predict(line_binned),linewidth=2,color='green',linestyle='-',label='linear regression')
ax2.plot(line,tree_.predict(line_binned),linewidth=2,color='red',linestyle=':',label='decision tree')
#绘制和箱宽一致的竖线
ax2.vlines(enc.bin_edges_[0],*plt.gca().get_ylim(),linewidth=1,alpha=0.2)#ax2.vlines(x轴，y轴)
#enc.bin_edges_返回的是分箱的上限和下限
#将原始数据分布绘制在图像上
ax2.plot(x[:,0],y,'o',c='k')
ax2.legend(loc='best')
ax2.set_xlabel('input feature')
ax2.set_title('result after discretization')
plt.tight_layout()
plt.show()

#分箱的箱数会影响模型的效果
#怎样选取最优的箱子
from sklearn.model_selection import cross_val_score
pred,score,var=[],[],[]
bins_range=[2,5,10,15,20,30]
for i in bins_range:
    enc=KBinsDiscretizer(n_bins=i,encode='onehot')
    x_binned=enc.fit_transform(x)
    line_binned=enc.fit_transform(line)
    linear=LinearRegression()
    #全数据集上的交叉验证
    cvresult=cross_val_score(linear,x_binned,y,cv=10)
    score.append(cvresult.mean())
    var.append(cvresult.var())
    #测试数据集上的打分结果
    pred.append(linear.fit(x_binned,y).score(line_binned,np.sin(line)))
#绘制图像
plt.figure()
plt.plot(bins_range,pred,c='orange',label='test')
plt.plot(bins_range,score,c='k',label='full data')
plt.plot(bins_range,score+np.array(var)*0.5,c='red',linestyle='--',label='var')
plt.plot(bins_range,score-np.array(var)*0.5,c='red',linestyle='--')
plt.legend()
plt.show()