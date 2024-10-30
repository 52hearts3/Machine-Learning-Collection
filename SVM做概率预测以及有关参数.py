#decision_function 返回的是每个样本点到超平面的距离，此接口返回的值我们叫做置信度
#不过，置信度始终不是概率，它没有边界，可以无限大，为了解决这个矛盾，SVC有重要参数probability

#参数  probability
#填布尔值，可不填，默认为False
#指是否启用概率估计，进行必须在fit之前调用它，启用此功能会减慢SVC的运算速度
#设置为True则会启用，启用之后，SVC的predict_proba和predict_log_proba则会生效
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
class_1=500#类别1有500个样本
class_2=50#类别2只有50个样本
centers=[[0.0,0.0],[2.0,2.0]]#设定两个类别的中心
clusters_std=[1.5,0.5]#设定两个类别的方差，一般来说，样本量比较大的类别会更加松散，方差更大
x,y=make_blobs(n_samples=[class_1,class_2],centers=centers,cluster_std=clusters_std,random_state=0,shuffle=False)
plt.scatter(x[:,0],x[:,1],c=y,cmap='rainbow')
plt.show()
clf_proba=SVC(kernel='linear',C=1.0,probability=True)
clf_proba.fit(x,y)
score=clf_proba.predict_proba(x)
print(score)#生成的每个样本被预测到每个标签的概率
print(score.shape)#(550, 2) 其中2指两个标签
distance=clf_proba.decision_function(x)
print(distance)#生成的是每个点到决策边界的距离
print(distance.shape)#(550,)
#SVC的概率分析在大型数据集上非常昂贵，计算会非常缓慢
#另外，由于platt缩放的理论原因，在二分类过程中，可能会出现predict_proba返回的概率小于0.5，但样本依然被标记为正类的情况出现
#所以如果我们确实需要置信度分数，我们可以将probability设置为False，使用decision_function这个接口而不是predict_proba