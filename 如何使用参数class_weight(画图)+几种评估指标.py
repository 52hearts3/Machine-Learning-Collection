import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
#创建样本不均衡的数据集
class_1=500  #多数类为比例10
class_2=50   #少数类比例为1
centers=[[0.0,0.0],[2.0,2.0]]#设定两个类别的中心
clusters_std=[1.5,0.5]#设置两个类别的方差，通常来说，样本量比较大的类别会更加松散
x,y=make_blobs(n_samples=[class_1,class_2],centers=centers,cluster_std=clusters_std,random_state=0,shuffle=False)
plt.scatter(x[:,0],x[:,1],c=y,cmap='rainbow',s=10)#多数类y为0  少数类y为1
plt.show() #其中红色点为少数类，紫色点为多数类
#建模
#不设定class_weight
clf=SVC(kernel='linear',C=1)
clf.fit(x,y)
score=clf.score(x,y)
print(score)
#设定class_weight
w_clf=SVC(kernel='linear',C=1,class_weight={0:1,1:10})#标签为0，1  #标签为1的占少数，为了尽可能的捕捉它，我们要增大它的权重
w_clf.fit(x,y)
score=w_clf.score(x,y)
print(score)
#没有做均衡的准确率比做均衡的准确率高
#做样本均衡后准确率下降

#绘制两个模型的决策边界
plt.figure(figsize=(6,5))
plt.scatter(x[:,0],x[:,1],c=y,cmap='rainbow',s=10)
ax=plt.gca()#创建子图
#绘制决策边界第一步  要有网格
x_lim=ax.get_xlim()
y_lim=ax.get_ylim()
xx=np.linspace(x_lim[0],x_lim[1],30)
yy=np.linspace(y_lim[0],y_lim[1],30)
YY,XX=np.meshgrid(yy,xx)
xy=np.vstack([XX.ravel(),YY.ravel()]).T
#找出决策边界到样本点的距离
z_clf=clf.decision_function(xy).reshape(XX.shape)
a=ax.contour(XX,YY,z_clf,colors='black',levels=[0],alpha=0.5,linestyles=['-'])
z_w_clf=w_clf.decision_function(xy).reshape(XX.shape)
b=ax.contour(XX,YY,z_w_clf,colors='red',levels=[0],alpha=0.5,linestyles=['-'])
#画图例
plt.legend([a.collections[0],b.collections[0]],['non weight','weighted'],loc='upper right')
#plt,legend[[对象列表],[图例列表],loc]
plt.show()
#可以看出，从准确路角度来看，不做样本平衡的时候准确率反而更高，做了样本平衡准确率反而降低了
#这是因为做了样本平衡后，为了更加有效的捕捉出少数类，模型误伤了许多多数类样本，而多数类被分错的样本数量大于少数类被分对的样本数量，使模型的整体准确率下降
#如果我们是以模型整体的准确率为目的，那我们就不要设置class_weight
#但在现实生活中，我们往往在追求捕捉少数类，比如识别出潜在犯罪者，金融评分卡等，这时要设置class_weight

#除了score外还有其他评分指标
#精确度  当在意精确率时使用 （判错少数类的考量）
#所有判断正确并确实为1的样本/所有被判为1的样本   （在此案例中，少数类的y为1）
#对于没有class_weight来说
score_1=(y[y==clf.predict(x)]==1).sum()/(clf.predict(x)==1).sum()#y[y==clf.predict(x)] 所有为True的y会被切片出来
#对于有class_weight来说
score_2=(y[y==w_clf.predict(x)]==1).sum()/(w_clf.predict(x)==1).sum()
print(score_1,score_2)

#敏感度（recall 召回率）  在意是否捕捉了全部少数类  （判错少数类的考量）
#所有预测为1的点/全部为1的点  （在此案例中，少数类的y为1）
#没有class_weight
score_1=(y[y==clf.predict(x)]==1).sum()/(y==1).sum()
#有class_weight
score_2=(y[y==w_clf.predict(x)]==1).sum()/(y==1).sum()
print(score_1,score_2)

#判错多数类的考量。特异度与假正率
#特异度   所有被正确预测为0的样本/所有0的样本
#没有class_weight
score_1=(y[y==clf.predict(x)]==0).sum()/(y==0).sum()
#有class_weight
score_2=(y[y==w_clf.predict(x)]==0).sum()/(y==0).sum()
print(score_1,score_2)
#假正率  1-特异度  衡量一个模型将多数类判断错误的指标