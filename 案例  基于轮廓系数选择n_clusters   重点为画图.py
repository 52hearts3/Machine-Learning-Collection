from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm      #cm为colormap
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs  #make_blobs可以理解为帮我做几个簇
x,y=make_blobs(n_samples=500,n_features=2,centers=4,random_state=1)#n_features=2特征为2  centers=4  4个簇   y为标签
#基于轮廓系数选择n_clusters
#画两个图   1  每一个聚出来的类的轮廓系数是多少，各个类之间的轮廓系数的对比
# 2  聚类完毕后图像的分布是什么样的
n_clusters=4
#创建一个画布，画布上有一行两列两个图
fig,(ax1,ax2)=plt.subplots(1,2)
#画布尺寸
fig.set_size_inches(18,7)
#第一个图是我们的轮廓系数图像，是由各个簇的轮廓系数组成的横向条形图
#横向条形图的横坐标是我们的轮廓系数取值，纵坐标是我们的样本，因为轮廓系数是对于每一个样本进行计算的

#首先来设定横坐标
#轮廓系数的取值范围在-1到1之间，但我们至少是希望轮廓系数要大于0的
#太长的横坐标不利于我们的可视化，所以只设定x轴的取值在-0.1到1之间
ax1.set_xlim([-0.1,1])
#接下来调整纵坐标  纵坐标是样本个数
#通常来说，纵坐标是从0开始，最大值为x.shape[0]
#但我们希望，每个簇能够排在一起，不同的簇之间能够有一定的空隙
#以使我们看到不同的条形图聚合成的块，理解它是对应了哪个簇
#因此我们在设定纵坐标的取值范围时，在x.shapr[0]上，加一个距离(n_clusters+1)*10.留作间隔用
ax1.set_ylim([0,x.shape[0]+(n_clusters+1)*10])

#开始建模
clusters=KMeans(n_clusters=n_clusters,random_state=10)
clusters.fit(x)
cluster_labels=clusters.labels_
#调用轮廓系数，注意，silhouette_score生成的是所有样本点的轮廓系数的均值
#两个需要输入的参数是  特征矩阵x 和聚类完毕后的标签
silhouette_ave=silhouette_score(x,cluster_labels)
#用print来报一下结果，现在的簇数量n_clusters下，整体的轮廓系数究竟是多少
print('for n_clusters =',n_clusters,'the average silhouette_score is:',silhouette_ave)
#调用silhouette_samples,返回每个样本点的轮廓系数，这就是我们的横坐标
sample_silhouette_values=silhouette_samples(x,cluster_labels)

#开始画图
#设定y轴的初始取值
y_lower=10
#接下来，对每一个簇进行循环
for i in range(n_clusters):
    #从每个样本的轮廓系数结果中抽取第i个簇的轮廓系数
    ith_cluster_silhouette_values=sample_silhouette_values[cluster_labels==i]
    #对它进行排序，注意 sort()这个命令会直接改掉原数据的顺序
    ith_cluster_silhouette_values.sort()
    #查看这一个簇中究竟有多少样本你
    size_cluster_i=ith_cluster_silhouette_values.shape[0]
    #这一个簇的y轴上的取值，应该是由初始值(y_lower)开始，到初始值加上这个簇中的样本数量结束(y_upper)
    y_upper=y_lower+size_cluster_i
    #在colormap中，使用小数来调颜色的函数
    #在nipy_spectral([输入任意小数来代表一个颜色])
    #在这里我们希望每个簇的颜色是不同的，我们需要的颜色种类刚好是循环的个数
    #在这里，我们只要能够确保，每次循环生成的小数是不同的，可以使用任意方法来获取小数
    #在这里 ，使用i的浮点数/n_clusters,在不同的i下，自然生成不同的小数
    color=cm.nipy_spectral(float(i)/n_clusters)
    #开始填充子图1中的内容
    #fill_between是一个让范围中的柱状图都统一颜色的函数
    #fill_betweenx的范围是在纵坐标上
    #fill_betweeny的取值是在横坐标上
    #fill_betweenx的参数应该输入(纵坐标的下限，纵坐标的上限，x轴的取值，柱状图的颜色)
    ax1.fill_betweenx(np.arange(y_lower,y_upper),ith_cluster_silhouette_values,facecolor=color,alpha=0.7)
    #为每个簇的轮廓系数写上簇的编号，并且让簇的编号显示在坐标轴上的每个条形图的中间位置
    #text的参数为(要显示编号的位置的横坐标，要显示编号位置的纵坐标，要显示的编号内容)
    ax1.text(-0.05,y_lower+0.5*size_cluster_i,str(i))
    #为下一个簇计算新的y轴上的初始值，是每一次迭代之后，y的上限再加10
    #以此来保证，不同的簇的图像之间显示空隙
    y_lower=y_upper+10
#给图1加上标题
ax1.set_title('teh silhouette plot for the various clusters')
ax1.set_xlabel('the silhouette coefficient values')
ax1.set_ylabel('cluster label')
#把整个数据集上的轮廓系数的均值以虚线的形式放入我们的图中
ax1.axvline(x=silhouette_ave,color='red',linestyle='--')
#让y轴不在显示任何刻度
ax1.set_yticks([])
#让x轴显示我们规定的刻度
ax1.set_xticks([-0.1,0,0.2,0.4,0.6,0.8,1])
#plt.show()

#开始对第二个图进行处理，首先获取颜色，由于这里没有循环，因此我们需要一次性生成多个小数来获取多个颜色
colors=cm.nipy_spectral(cluster_labels.astype(float)/n_clusters)
ax2.scatter(x[:,0],x[:,1],marker='o',s=8,c=colors)
#把生成的质心放在图像中
centers=clusters.cluster_centers_
ax2.scatter(centers[:,0],centers[:,1],marker='x',c='red',alpha=1,s=200)
#为图2设置标题
ax2.set_title('the visualization of the clustered data')
ax2.set_xlabel('feature space for the 1st feature')
ax2.set_ylabel('feature space for the 2nd feature')
#为整个图设置标题
plt.suptitle(('with n_clusters=%d'% n_clusters))
plt.show()
#即可以选轮廓系数最高的，又可以选每一簇都对轮廓系数有较高贡献的
#到时候直接拿来用就行