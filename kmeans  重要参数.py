#n_clusters  是kmeans中的k  #默认为8
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
#重要属性labels_   查看聚好的类别，每个样本对应的类
y_pred=cluster.labels_
print(y_pred)
#cluster.fit_predict(x)  的结果与y_pred的结果一样，都是查看聚类后每个数据对应的簇
#当数据量非常大时我们可以先切片为训练集和测试集，用训练集进行fit 之后用x去predict得到大致结果
#如
cluster_smallsub=KMeans(n_clusters=n_clusters,random_state=0).fit(x[:200])
y_pred_=cluster_smallsub.predict(x)
print(y_pred_==y_pred)

#重要属性cluster_centers_  查看质心
print(cluster.cluster_centers_)
#查看最佳迭代次数
print(cluster.n_iter_)
#重要属性inertia_  查看总距离平方和
print(cluster.inertia_)
centroid=cluster.cluster_centers_
#对聚类数据可视化
color=['red','pink','orange','gray']
fig,ax1=plt.subplots(1)
for i in range(n_clusters):
    ax1.scatter(x[y_pred==i,0],x[y_pred==i,1],marker='o',s=8,color=color[i])
ax1.scatter(centroid[:,0],centroid[:,1],marker='x',s=15,c='black')
plt.show()
#随着n_cluster越来越大  inertia_会越来越小  所以只看inertia_评估聚类是不行的