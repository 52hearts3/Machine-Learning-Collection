#max_iter  默认300
#单次运行的k-means的最大迭代次数

#tol  浮点数  默认1e-4
#两次迭代inertia下降的量，如果两次迭代之间的inertia下降的值小于tol设定的值，迭代就会停下

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm      #cm为colormap
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs  #make_blobs可以理解为帮我做几个簇
x,y=make_blobs(n_samples=500,n_features=2,centers=4,random_state=1)#n_features=2特征为2  centers=4  4个簇   y为标签
random=KMeans(n_clusters=10,init='random',max_iter=10,random_state=420)
random.fit(x)
y_pred_max_10=random.labels_
score=silhouette_score(x,y_pred_max_10)
print(score)
random=KMeans(n_clusters=10,init='random',max_iter=20,random_state=420)
random.fit(x)
y_pred_max_20=random.labels_
score=silhouette_score(x,y_pred_max_20)
print(score)#可以看到迭代10次比20次分数高