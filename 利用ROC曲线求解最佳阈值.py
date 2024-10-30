#我们希望模型在捕获少数类的能力变强时，尽量不误伤多数类，也就是说，随着recall的变化，FPR的大小以小越好
#我们只需找到recall和FPR差距最大的点，这个点，又叫做约登指数
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
#我们这里的thresholds已经不是概率了，因为我们的decision_function传入的是距离
#使用AUC计算面积
from sklearn.metrics import roc_auc_score
s=roc_auc_score(y,clf.decision_function(x))
print(s)

#recall和FPR的差距 recall-FPR
#找到最佳threshold
max_index=(recall-FPR).tolist().index(max(recall-FPR))
best_threshold=thresholds[max_index]
print(best_threshold)#对于decision_function来说
#-1.0861521615211185  样本到超平面的距离大于它判对，小于它判错，距离的正负代表在超平面的上方还是下方
#在图上看看在哪里
#画图
plt.figure(figsize=(20,5))
plt.scatter(FPR[max_index],recall[max_index],c='black',s=30)
plt.plot(FPR,recall,color='red',label='ROC curve(s=%0.2f)'%s)
plt.plot([0,1],[0,1],color='black',linestyle='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('false positive rate')
plt.ylabel('recall')
plt.title('receiver operating characteristic example')
plt.legend(loc='lower right')
plt.show()