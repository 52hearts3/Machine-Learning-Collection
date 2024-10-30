#我们使用模块metrics中的类silhouette_score来计算轮廓系数，它返回的是一个数据的平均轮廓系数
#silhouette_samples返回的是每个样本自己的轮廓系数
from sklearn.datasets import make_blobs  #make_blobs可以理解为帮我做几个簇
import matplotlib.pyplot as plt
#自己创造数据集
x,y=make_blobs(n_samples=500,n_features=2,centers=4,random_state=1)#n_features=2特征为2  centers=4  4个簇   y为标签
fig,ax1=plt.subplots(1)
ax1.scatter(x[:,0],x[:,1],marker='o',s=8)#marker 点的形状   s=8 点的大小
plt.show()
from sklearn.cluster import KMeans
n_clusters=4
cluster=KMeans(n_clusters=n_clusters,random_state=0).fit(x)
y_pred=cluster.labels_
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
score_1=silhouette_score(x,y_pred)#分4簇比分3簇要好
print(score_1)
score_2=silhouette_samples(x,y_pred)
print(score_2.shape)#(500,)