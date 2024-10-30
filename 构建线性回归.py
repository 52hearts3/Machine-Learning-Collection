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
print(y_hat.max())#7.146198214270862
#由此可见模型的预测效果并不太好
print(reg.coef_)
print(reg.intercept_)

print([*zip(df.columns,reg.coef_)])