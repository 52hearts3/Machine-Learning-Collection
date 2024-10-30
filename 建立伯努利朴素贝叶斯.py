import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss
class_1=500
class_2=500#两个类别分别设定500个样本
centers=[[0.0,0.0],[2.0,2.0]]#设定两个类别的中心
clusters_std=[0.5,0.5]#设定两个类别的方差
x,y=make_blobs(n_samples=[class_1,class_2],centers=centers,cluster_std=clusters_std,random_state=0,shuffle=False)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=420)

#归一化，防止训练集和测试集出现负数
mms=MinMaxScaler()
mms.fit(x_train)
x_train_=mms.transform(x_train)
x_test_=mms.transform(x_test)
from sklearn.naive_bayes import BernoulliNB

#不设置二值化
bnl_=BernoulliNB()
bnl_.fit(x_train_,y_train)
print(bnl_.score(x_test_,y_test))
print(brier_score_loss(y_test,bnl_.predict_proba(x_test_)[:,1],pos_label=1))

#设置二值化的阈值为0.5
bnl_=BernoulliNB(binarize=0.5)
bnl_.fit(x_train_,y_train)
print(bnl_.score(x_test_,y_test))
print(brier_score_loss(y_test,bnl_.predict_proba(x_test_)[:,1],pos_label=1))