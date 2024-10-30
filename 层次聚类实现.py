from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
x,y=make_blobs(n_samples=500,n_features=2,centers=4,random_state=1)#n_features=2特征为2  centers=4  4个簇   y为标签
#model=AgglomerativeClustering(n_clusters=3,metric='euclidean',linkage='complete',compute_full_tree=True)
model=AgglomerativeClustering(n_clusters=None,compute_full_tree=True,distance_threshold=0)
#只有当compute_full_tree=True的时候距离才会被计算
#linkage可取值
#ward  最小方差
#complete  最大距离
#average  平均距离
#single  最小距离

#metric默认欧几里得距离
model.fit(x)
#层次聚类没有中心的概念
print('每个样本所在的簇',model.labels_)
import matplotlib.pyplot as plt
color=['g','r','b']
print(x.shape)
#for i in range(x.shape[0]):
    #plt.scatter(x[i,0],x[i,1],c=color[model.labels_[i]],s=20)
#plt.show()
print(model.children_)
print(model.distances_)
from scipy.cluster.hierarchy import dendrogram
import numpy as np

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
plot_dendrogram(model, truncate_mode="level", p=5)
plt.show()