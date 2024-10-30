from sklearn.datasets import make_circles
x,y=make_circles(100,factor=0.1,noise=0.1)
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='rainbow')
plt.show()
#使用线性的方法来决策效果会很差
#明显。现在的SVM已经不再适用于我们现在的情况了，我们无法找到一条直线来划分我们的数据集，让直线的两边分别是两种数据
#这个时候，如果我们在原本的x和y的基础上，添加一个维度r，变为三维，我们可以解决这个问题
#包装画图函数
def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax=plt.gca()
    x_lim=ax.get_xlim()#取出x y轴的最大值和最小值
    y_lim=ax.get_ylim()
    x=np.linspace(x_lim[0],x_lim[1],30)
    y=np.linspace(y_lim[0],y_lim[1],30)
    y,x=np.meshgrid(y,x)
    xy=np.vstack([x.ravel(),y.ravel()]).T
    p=model.decision_function(xy).reshape(x.shape)
    ax.contour(x,y,p,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
#实例化+训练模型
clf=SVC(kernel='rbf')
clf.fit(x,y)
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='rainbow')
plot_svc_decision_function(clf)
plt.show()
r=np.exp(-(x**2).sum(1))
r_lim=np.linspace(min(r),max(r),100)
#绘制三维图像
def plot_3D(elev=30,azim=30,x=x,y=y):
    #elev为上下旋转的角度
    #azim表示平行旋转的角度
    ax=plt.subplot(projection='3d')
    ax.scatter3D(x[:,0],x[:,1],r,c=y,s=50,cmap='rainbow')
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    plt.show()
#plot_3D()
#我们刚才做的，计算r，并将r作为数据的第三维度来将数据升维的过程，成为核变换
#即是将数据投影在高维空间中，以寻找能将数据完美分割的超平面