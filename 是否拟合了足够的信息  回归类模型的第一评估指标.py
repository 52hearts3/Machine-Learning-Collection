#对于回归类算法而言，只探索数据预测是否准确是不足够的，除了数据本身的数值大小之外，我们还希望模型能够捕捉到数据的规律
#比如数据的分布规律，单调性等等，而是否捕捉到了这些信息无法使用mse表示
#我们用R**2衡量模型捕捉到的信息量
#衡量的是1-我们的模型没有捕获到的信息占真实标签中所带的信息量的比例，所以R**2越接近1越好

#R**2是回归类模型的第一评估指标！！！

#R**2可以使用三种方式调用
# 1 直接从metrics调用r2_score
# 2 从线性回归LinearRegression的接口score来调用
#所有回归类算法的score默认的是R**2
# 3 在交叉验证中，scoring输入r2来调用

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing
import pandas as pd
house=fetch_california_housing()
data=house.data
target=house.target
feature_names=house.feature_names
df=pd.DataFrame(data=data,columns=feature_names)
print(df.head())
x_train,x_test,y_train,y_test=train_test_split(df,target,test_size=0.3,random_state=420)
#重设索引
for i in [x_train,x_test]:
    i.index=range(i.shape[0])
#进行标准化
#先用训练集训练标准化的类，然后再用训练好的类分别转化训练集和测试集
from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
sd.fit(x_train)
x_train=sd.transform(x_train)
x_test=sd.transform(x_test)
#建模
reg=LinearRegression()
reg.fit(x_train,y_train)
y_hat=reg.predict(x_test)#预测y
#调用类
from sklearn.metrics import r2_score
print(r2_score(y_hat,y_test))#输入预测值，真实值
#通过score调用
print(reg.score(x_test,y_test))#输入x，y

#为什么二者评估指标相同，结果却不同
#参数填入反了
#所以我们在使用metrcis模块来评估模型时，必须要检查清楚指标要求我们先输入真实值还是预测值
#使用ctrl+p键来查看究竟哪个值先输入
#可以看到，真实值在前，预测值在后
print(r2_score(y_test,y_hat))
#也可以直接指定参数，就不必在意顺序了
print(r2_score(y_true=y_test,y_pred=y_hat))

#使用交叉验证调用
print(cross_val_score(reg,data,target,cv=10,scoring='r2').mean())

#我们可以画图看看数据的分布
import matplotlib.pyplot as plt
plt.plot(range(len(y_test)),sorted(y_test),c='black',label='hat')
plt.plot(range(len(y_hat)),sorted(y_hat),c='red',label='predict')
plt.legend()
plt.show()