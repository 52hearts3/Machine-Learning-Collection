
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
wine=load_wine()
data=wine.data
target=wine.target
feature_names=wine.feature_names
df=pd.DataFrame(data=data,columns=feature_names)
print(df)
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.3)
rfc=RandomForestClassifier(random_state=0)
rfc.fit(x_train,y_train)
#estimators_ 随机森林的重要属性之一，查看森林中树的状况
print(rfc.estimators_)
print(rfc.estimators_[0].random_state)
#查看所有树的random_state
for i in range(len(rfc.estimators_)):
    print(rfc.estimators_[i].random_state)
#也就是即使在训练时加了随机种子，随机森林训练时每个树的random_state是不同的

#boostrap参数默认为true 代表采用这种有放回的随机抽样技术，通常，这个参数不会被设置为false
#然而这种放回抽样有自己的问题，当n足够大时，会有37%的训练数据会被浪费掉，没有参与建模，这些数据被成为袋外数据（obb）
#n为训练集样本个数
#也就是说，当使用随机森林时，我们可以不划分训练集和测试集，只需要用袋外数据来测试我们的模型即可
#但是，当n和n_estimators都不够大时，很可能没有数据掉落在袋外，自然就不能使用

#如果希望用袋外数据来测试，就在实例化时把obb_score这个参数调整为true
#我们可以使用obb_score_来查看在袋外数据的测试结果
rfc=RandomForestClassifier(oob_score=True)
rfc.fit(data,target)
print(rfc.oob_score_)

#重要属性和接口
#estimators_
#boostrap
#obb_score_
#feature_importances_

#常用接口
#apply
#fit
#predict
#score
print(rfc.score(x_test,y_test))
print(rfc.feature_importances_)#重要性越大，越重要
print(rfc.apply(x_test))#返回这个样本在的这一课树中所在的叶子节点的索引
print(rfc.predict(x_test))
print(rfc.predict_proba(x_test))#返回每一个样本被分到某一个标签的概率