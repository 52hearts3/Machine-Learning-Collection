#嵌入法是一种让算法自己决定使用哪些特征的方法，即特征选择和算法训练同时进行
#相比于过滤法，嵌入法会更加精确到模型本身，是过滤法的升级版
#嵌入法计算慢
#feature_selection.SelectModel
#SelectModel可以和带有惩罚项的算法使用，如随机森林和树模型具有属性feature_importances，逻辑回归（分类），线性支持向量机
#重要属性
#estimator  只要是带feature_importances_或者coef_属性，或者带有l1 l2惩罚项的模型都可以使用
#threshold  特征重要性的阈值，重要性低于这个阈值的特征都将被删除（0到1）
import pandas as pd
data=pd.read_csv(r'D:\game\sklearn\特征选择\digit recognizor.csv')
print(data.head())
x=data.iloc[:,1:]
y=data.iloc[:,0]
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
rfc=RandomForestClassifier(n_estimators=100,random_state=0)
x_embedded=SelectFromModel(rfc,threshold=0.005).fit_transform(x,y)##参数norm_order=1 使用l1范式进行筛选
#0.005对于这些特征来说是非常高的阈值，因为平均每个特征只能分到0.001
print(x_embedded.shape)#只剩43和特征
#如何确定threshold取多少合适  画学习曲线
import numpy as np
import matplotlib.pyplot as plt
rfc.fit(x,y)
threshold=np.linspace(0,(rfc.feature_importances_).max(),20)
#score=[]
#for i in threshold:
    #x_embedded=SelectFromModel(rfc,threshold=i).fit_transform(x,y)
    #once=cross_val_score(rfc,x_embedded,y,cv=5).mean()
    #score.append(once)
#plt.plot(threshold,score)
#plt.show()
#从图像上看，随着阈值越高，模型效果越差，但是在0.001之前，模型效果可以维持在0.93以上，我们可以1从中挑选一个数值验证效果
x_embedded=SelectFromModel(rfc,threshold=0.0005).fit_transform(x,y)
score=cross_val_score(rfc,x_embedded,y,cv=5).mean()
print(x_embedded.shape)
print(score)#0.9531666666666668