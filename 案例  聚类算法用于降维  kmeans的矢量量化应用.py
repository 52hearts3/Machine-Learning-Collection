#矢量量化的降维是在同等样本量上压缩信息的大小，不改变数据的维度
#只改变这些特征在样本上的信息量
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin  #对两个序列中的点进行距离匹配的函数
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle#洗牌    #打乱一个有序序列的函数  可以打乱dataframe
china=load_sample_image('china.jpg')
print(china.dtype)
print(china.shape)#(427, 640, 3)  长  宽  像素  3值三个特征，3个数决定的颜色
print(china[0][0])#[174 201 231]  3个特征决定一种颜色
#查看包含多少种不同的颜色
new_image=china.reshape((427*640,3))#只把特征做列
print(new_image.shape)
import pandas as pd
un=pd.DataFrame(new_image).drop_duplicates().shape  #drop_duplicates()去除重复值的函数
print(un)#(96615, 3)  有九万多种颜色，每个颜色由三个特征组成

#图像可视化
plt.figure(figsize=(15,15))
plt.imshow(china)#imshow  只接受三维数组形成的图片
plt.show()
#查看模块中的另一张照片
flower=load_sample_image('flower.jpg')
plt.figure(figsize=(15,15))
plt.imshow(flower)
plt.show()

#使用china的图片来进行矢量量化
#这个图片中有九万多个颜色，我们希望压缩到64种颜色，还不严重损耗图像的质量
#为此，我们需要使用kmeans来将九万种颜色聚类成64类，然后使用64簇的质心来代替全部的9w种颜色

#数据预处理  kmeans不接受三维数组作为特征矩阵
n_cluster=64
#数据归一化处理   plt.imshow在浮点数上表现优异，在这里我们把china中的数据转换为浮点数，压缩到0到1之间
china=np.array(china,dtype=np.float64)/china.max()
#把china从图像格式，转换为矩阵格式
w,h,d=original_shape=tuple(china.shape)
assert d==3
#assert  要求d必须等于3 不等于3就报错
image_array=np.reshape(china,(w*h,d))
print(image_array.shape)
#np.reshape(a,newshape,order='C')  第一个参数a是要改变结构的对象，第二个参数是要改变的新结构,
#详细见  疑难困惑  np.reshape效果

#开始建模
#因为行数太大，所以先使用1000个数据来找出质心
image_array_sample=shuffle(image_array,random_state=0)[:1000]
kmeans=KMeans(n_clusters=n_cluster,random_state=0)
kmeans.fit(image_array_sample)
center=kmeans.cluster_centers_
#找出质心后，按照已有的质心对所有数据进行聚类
labels=kmeans.predict(image_array)
print(set(labels))#集合set有去重效果  详见疑难困惑

#用质心替换掉所有的样本
image_kmeans=image_array.copy()
#labels是这27w个样本点所对应的簇的质心的索引
for i in range(w*h):
    image_kmeans[i]=center[labels[i]]
df=pd.DataFrame(image_kmeans).drop_duplicates().shape
print(df)
#恢复图片的结构
image_kmeans=image_kmeans.reshape((w,h,d))
print(image_kmeans.shape)

#对数据进行随机矢量量化，以对比kmeans的效果
centroid_random=shuffle(image_array,random_state=0)[:n_cluster]
labels_random=pairwise_distances_argmin(centroid_random,image_array,axis=0)
#函数pairwise_distances_argmin(x1,x2,axis)  x1和x2是序列
#用来计算x2中每个样本到x1中的每个样本点的距离，并返回和x2相同形状的，x2的每个点到x1的最近的点的索引
print(len(set(labels_random)))
image_random=image_array.copy()
#使用随机质心替换所有样本
for i in range(w*h):
    image_random[i]=centroid_random[labels_random[i]]
#恢复图片结构
image_random=image_random.reshape(w,h,d)

#画图
plt.figure(figsize=(10,10))
plt.axis('off')#不要显示坐标轴
plt.title('original image (96615 colors)')
plt.imshow(china)
plt.show()

plt.figure(figsize=(10,10))
plt.axis('off')#不要显示坐标轴
plt.title('quantized image (64 colors,kmeans)')
plt.imshow(image_kmeans)
plt.show()

plt.figure(figsize=(10,10))
plt.axis('off')#不要显示坐标轴
plt.title('quantized image (64 colors,random)')
plt.imshow(image_random)
plt.show()