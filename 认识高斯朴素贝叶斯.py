#从naive_bayes.GaussianNB调用

#参数
#prior  可以输入任何数组，形状为(n_classes).表示类的先验概率，如果指定，则不根据数据调整先验，如果不指定，则自行根据数据计算先验概率
#var_smoothing  填浮点数，可不填（默认1e-9），在估计方差时，为了追求估计的稳定性，将所有特征的方差中的最大的方差以某个比例添加到估计的方差中
#这个比例，由var_smoothing 控制

#通常我们什么都不填写
#由于贝叶斯没有太多的参数，因此贝叶斯的厂长空间并不大
#如果贝叶斯算法的效果不是太理想，我们一般会考虑换模型

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits=load_digits()
x,y=digits.data,digits.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=420)
print(x_train.shape)
print(np.unique(y_train))#多分类问题
gnb=GaussianNB()
gnb.fit(x_train,y_train)
acc_score=gnb.score(x_test,y_test)
print(acc_score)
#查看预测结果
y_pred=gnb.predict(x_test)
#查看预测的概率结果
prob=gnb.predict_proba(x_test)
print(prob.shape)#每一列对应一个标签类别下的概率
print(prob.sum(axis=1))#每一行的和都为1

#使用混淆矩阵查看分类结果
from sklearn.metrics import confusion_matrix
me=confusion_matrix(y_test,y_pred)
print(me)

#roc曲线不能用于多分类