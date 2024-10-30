from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
x,y=make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.6)#cluster_std=0.6  指簇的方差为0.6
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='rainbow')#c=y  颜色为y
plt.xticks([])
plt.yticks([])
plt.show()

#画决策边界
#contour  专门画等高线的函数
#参数   x,y  二维平面上所有的点的横纵坐标取值，选填，一般要求二维结构
#不填默认x=range(z.shape[1]),y=range(z.shape[0])
#  z  必填  平面上所有点对应的高度  结构与x，y保持一致
#levels  可不填，不填默认显示所有等高线，填写用于确定等高线的数量和位置
#如果填写整数n，则显示n个数据区间，即绘制n+1条等高线，水平高度自动选择
#如果填写的是数组或列表，则在指定的高度级别绘制等高线，列表或数组中的值必须按递增顺序排序

#其实，我们只需把所有到决策边界的距离为0的点相连，就是我们的决策边界
#把所有到决策边界的相对距离为1的点相连，就是我们的两个平行于决策边界的超平面
#此时，我们输入的高度z就是平面上的任意点到达超平面的距离

# 画决策边界  1  首先绘制散点图
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='rainbow')
ax=plt.gca()#获取当前的子图，如果没有，则创建新的子图
#画决策边界  2  制作网格 meshgird
#获取平面上两条坐标轴的最大值和最小值
x_lim=ax.get_xlim()
y_lim=ax.get_ylim()
#在最大值和最小值之间形成30个有规律的数据
axis_x=np.linspace(x_lim[0],x_lim[1],30)
axis_y=np.linspace(y_lim[0],y_lim[1],30)
axis_y,axis_x=np.meshgrid(axis_y,axis_x)
print(axis_x.shape,axis_y.shape)  #(30, 30)
#使用meshgrid函数将两个一维向量转换为网格状的特征矩阵
#我们将使用这里形成的二维数组作为我们的contour函数中的x，y
xy=np.vstack([axis_x.ravel(),axis_y.ravel()]).T
#vstack能够将多个结构一致的一维数组按行堆叠起来
#vstack与meshgrid的使用见疑难困惑
print(xy.shape)
plt.scatter(xy[:,0],xy[:,1],s=1)
#plt.show()

#  3  建模，计算决策边界并找出网格上每个点到决策边界的距离
clf=SVC(kernel='linear')
clf.fit(x,y)
z=clf.decision_function(xy).reshape(axis_x.shape)
#重要接口decision_function  ,返回每个输入的样本对应的到决策边界的距离
#然后再将这个距离转换为axis_x的结构，这是由于画图的函数contour要求z的结构必须与x，y保持一致
# z是xy中所有样本点到决策边界的距离

#画决策边界和平行于决策边界的超平面
ax.contour(axis_x,axis_y,z,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])#levels=[-1,0,1]指画三条等高线，分别是z=-1，0，1
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.show()

print(clf.predict(x))
#根据决策边界，对x中的样本进行分类，
print(clf.score(x,y))
#返回给定测试数据和标签的准确度
print(clf.support_vectors_)
#返回支持向量（决策边界上下平移后的超平面穿过的点）
print(clf.n_support_)
#返回每个类中的支持向量的个数