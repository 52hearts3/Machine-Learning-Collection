import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.datasets import make_circles,make_moons,make_blobs,make_classification

#创建不同的数据集
n_samples=100
datasets=[make_moons(n_samples=n_samples,noise=0.2,random_state=0),
          make_circles(n_samples=n_samples,noise=0.2,factor=0.5,random_state=1),
          make_blobs(n_samples=n_samples,centers=2,random_state=5),
          make_classification(n_samples=n_samples,n_features=2,n_informative=2,n_redundant=0,random_state=5)]#每个创建出来的数据集都包含两个参数 x y
#定义不同核函数
Kernel=['linear','poly','rbf','sigmoid或logistic函数']
#四个数据集长什么样
for x,y in datasets:
    plt.figure(figsize=(5,4))
    plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='rainbow')
    plt.show()
#构建子图
n_rows=len(datasets)
n_cols=len(Kernel)+1
fig,axes=plt.subplots(n_rows,n_cols,figsize=(20,16))
#enumerate 结构  [(索引,内容)]
#第一层循环  在不同的数据集上循环
for ds_cnt,(x,y) in enumerate(datasets):
    #在图像的第一列放置原始数据
    ax=axes[ds_cnt,0]
    if ds_cnt==0:
        ax.set_title('input data')
    ax.scatter(x[:,0],x[:,1],c=y,zorder=10,cmap='rainbow',edgecolors='k')#zorder的值越大，图像层级就越大，即一张画布中有多个图像，这个图像的显示度就越高
    ax.set_xticks(())
    ax.set_yticks(())
    #第二层循环，在四种核函数的循环
    for est_idx,kernel in enumerate(Kernel):
        #定义子图位置
        ax=axes[ds_cnt,est_idx+1]
        #建模
        clf=SVC(kernel=kernel,gamma=2)
        clf.fit(x,y)
        score=clf.score(x,y)
        #绘制图像本身分布的散点图
        ax.scatter(x[:,0],x[:,1],c=y,zorder=10,cmap='rainbow',edgecolors='k')
        #绘制支持向量
        ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=50,facecolors='none',zorder=10,edgecolors='white')#facecolors='none' 画为透明的
        # clf.support_vectors_  返回支持向量（决策边界上下平移后的超平面穿过的点)
        #绘制决策边界
        x_min,x_max=x[:,0].min()-0.5,x[:,0].max()+0.5
        y_min,y_max=x[:,1].min()-0.5,x[:,1].max()+0.5
        #np.mgrid 合并了我们之前使用的np.linspace和np.meshgrid的用法
        #一次性使用最大值和最小值来生成网格
        #表示为[起始值:结束值:步长]  如果步长后面加j 如200j，表示取到200（包含200）
        xx,yy=np.mgrid[x_min:x_max:200j,y_min:y_max:200j]
        #np.c_ 类似于np.vstack的功能
        z=clf.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
        # 重要接口decision_function  ,返回每个输入的样本对应的到决策边界的距离
        #填充等高线不同区域的颜色
        ax.pcolormesh(xx,yy,z>0,cmap='rainbow')
        #填充等高线
        ax.contour(xx,yy,z,colors=['k','k','k'],linestyles=['--','-','--'],levels=[-1,0,1])
        #设定坐标轴为不显示
        ax.set_xticks(())
        ax.set_yticks(())
        #将标题放在第一行顶上
        if ds_cnt==0:
            ax.set_title(kernel)
        #为每张图添加分类的分数
        ax.text(0.95,0.06,('%.2f'%score).lstrip('0'),size=10,bbox=dict(boxstyle='round',alpha=0.8,facecolor='white'),
                transform=ax.transAxes,horizontalalignment='right')
        #0.95,0.06 添加文字的横纵坐标的位置
        #.lstrip('0') 表示不要显示0  bbox为分数添加一个白色的格子作为底色
        #transform=ax.transAxes  确定文字的坐标轴，就是ax子图的本身
        #horizontalalignment  位于坐标轴什么方向
plt.tight_layout()#图像间隔紧缩
plt.show()

#可以看到，图中标白色的全是支持向量，所以我们可以有多个支持向量，只不过算法选择了效果最好的支持向量