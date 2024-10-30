from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
cancer=load_breast_cancer()
data=cancer.data
target=cancer.target
feature_names=cancer.feature_names
#乳腺癌数据有569条记录，30个特征，样本量太小，很容易过拟合
rfc=RandomForestClassifier(random_state=90)
score=cross_val_score(rfc,data,target,cv=10).mean()
print(score)
#调参 1 n_estimators
score_=[]
for i in range(0,200,10):
    rfc=RandomForestClassifier(n_estimators=i+1,random_state=90)
    score=cross_val_score(rfc,data,target,cv=10).mean()
    score_.append(score)
print(max(score_),score_.index(max(score_))*10+1)
plt.figure(figsize=(20,5))
plt.plot(range(1,201,10),score_)
plt.show()
#根据结果n_estimator=71左右最好
score_=[]#精细化
for i in range(65,75):
    rfc=RandomForestClassifier(n_estimators=i+1,random_state=90)
    score=cross_val_score(rfc,data,target,cv=10).mean()
    score_.append(score)
print(max(score_),[*range(65,75)][score_.index(max(score_))])
plt.figure(figsize=(20,5))
plt.plot(range(65,75),score_)
plt.show()#是72
#n_estimators=72最好
#为网格搜索做准备
#调整最大深度
param_grid={'max_depth':np.arange(1,20,1)}
#对于小型数据，可以采用1到20试探，对于大型数据，可以先从30到50试探（也许还不够）
rfc=RandomForestClassifier(n_estimators=72,random_state=90)
GS=GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data,target)
print(GS.best_params_)#max_depth=8
print(GS.best_score_)#分数上升  既然剪枝后分数上升，说明模型泛化误差在图像右边
#调min_samples_leaf
param_grid={'min_samples_leaf':np.arange(1,22,1)}
rfc=RandomForestClassifier(n_estimators=72,max_depth=8,random_state=90)
GS=GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data,target)
print(GS.best_params_)# 结果为1 就是默认值
print(GS.best_score_)
#调min_samples_split
param_grid={'min_samples_split':np.arange(2,22,1)}
rfc=RandomForestClassifier(n_estimators=72,max_depth=8,random_state=90)
GS=GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data,target)
print(GS.best_params_)# 结果为2 就是默认值
print(GS.best_score_)
#max_features是最大特征数开平方
#调msx_features的步骤省略