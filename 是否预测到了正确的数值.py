#我们使用均方误差来衡量我们的预测值与真实值之间的差异，越接近0越好

#在sklearn中，我们有两种方式调用这个评估指标
#一种是使用sklearn专用模型评估模块metrics里的类mean_squared_error
#另一种是调用交叉验证的类cross_var_score并使用里面的scoring参数来设置均方误差
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
from sklearn.metrics import mean_squared_error
#直接调用
print(mean_squared_error(y_hat,y_test))
#0.53
print(y_test.mean())
#2.08
#也就是说预测错误了百分之二十左右
#交叉验证
print(cross_val_score(reg,data,target,cv=10,scoring='neg_mean_squared_error').mean())