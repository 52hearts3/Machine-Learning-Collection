#在sklearn中，我们用帮助我们计算ROC曲线的横坐标假正率FPR，纵坐标recall和对应的阈值的类sklearn.metrics.roc_curve
#同时有计算AUC面积的类sklearn.metrics.auc_score

#sklearn.metrics.roc_curve(y_true,y_score,pos_label,sample_weight,drop_intermediate)
#y_true 真实标签
#y_score 置信度分数，可以是正类样本的概率值，或置信度分数，或者decision_function返回的距离
#pos_label 输入整数或者字符串，默认none 表示被认为是正类样本的类别
#sample_weight 可不填，表示样本的权重
#drop_intermediate 填布尔值，默认true  如果设置为true，则会舍弃一些ROC曲线上不显示的阈值点，这对于计算一个比较轻量的ROC曲线非常有用
#此类返回 FPR recall以及阈值

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
class_1=500  #多数类为比例10
class_2=50  #少数类比例为1
centers=[[0.0,0.0],[2.0,2.0]]#设定两个类别的中心
clusters_std=[1.5,0.5]#设置两个类别的方差，通常来说，样本量比较大的类别会更加松散
x,y=make_blobs(n_samples=[class_1,class_2],centers=centers,cluster_std=clusters_std,random_state=0,shuffle=False)
plt.scatter(x[:,0],x[:,1],c=y,cmap='rainbow',s=10)#多数类y为0  少数类y为1
plt.show() #其中红色点为少数类，紫色点为多数类
#建模
#不设定class_weight
clf=SVC(kernel='linear',C=1)
clf.fit(x,y)
from sklearn.metrics import roc_curve
FPR,recall,thresholds=roc_curve(y,clf.decision_function(x),pos_label=1)
print(FPR.shape)
print(recall.shape)
print(thresholds.shape)
#我们这里的thresholds已经不是概率了，因为我们的decision_function传入的是距离

#使用AUC计算面积
from sklearn.metrics import roc_auc_score
s=roc_auc_score(y,clf.decision_function(x))
print(s)

#画图
plt.figure(figsize=(20,5))
plt.plot(FPR,recall,color='red',label='ROC curve(s=%0.2f)'%s)
plt.plot([0,1],[0,1],color='black',linestyle='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('false positive rate')
plt.ylabel('recall')
plt.title('receiver operating characteristic example')
plt.legend(loc='lower right')
plt.show()