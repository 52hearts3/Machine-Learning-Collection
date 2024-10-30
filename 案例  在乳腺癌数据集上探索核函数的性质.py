from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
cancer=load_breast_cancer()
data=cancer.data
target=cancer.target
print(data.shape)
print(np.unique(target))
#降为2维后可视化
from sklearn.decomposition import PCA
x_dr=PCA(2).fit_transform(data)
plt.scatter(x_dr[:,0],x_dr[:,1],c=target)
plt.show()
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.3,random_state=420)
Kernel=['linear','poly','rbf','sigmoid或logistic函数']
#for kernel in Kernel:
    #time0=time()
    #clf=SVC(kernel=kernel,gamma='auto',cache_size=5000)#cache_size=5000 表示允许使用多大的内存进行计算,单位mb
    #clf.fit(x_train,y_train)
    #print('the accuracy under kernel %s is %f'%(kernel,clf.score(x_test,y_test)))
    #print(datetime.datetime.fromtimestamp(time()-time0).strftime('%M:%S:%f'))
#运行非常非常慢
#缓慢的原因是卡在poly（多项式核函数）里，因为多项式核函数的degree默认为3，太高了，在三十个特征上计算的会很慢
#先运行其他三个
Kernel=['linear','rbf','sigmoid或logistic函数']
for kernel in Kernel:
    time0=time()
    clf=SVC(kernel=kernel,gamma='auto',cache_size=5000)#cache_size=5000 表示允许使用多大的内存进行计算,单位mb
    clf.fit(x_train,y_train)
    print('the accuracy under kernel %s is %f'%(kernel,clf.score(x_test,y_test)))
    print(datetime.datetime.fromtimestamp(time()-time0).strftime('%M:%S:%f'))
#得出结果  乳腺癌数据集是线性数据集，
#如果数据是线性的，那如果我们把degree调为1 多项式核函数应该也可以得到不错的效果
print('*'*60)
Kernel=['linear','poly','rbf','sigmoid或logistic函数']
for kernel in Kernel:
    time0=time()
    clf=SVC(kernel=kernel,gamma='auto',cache_size=5000,degree=1)#cache_size=5000 表示允许使用多大的内存进行计算,单位mb
    clf.fit(x_train,y_train)
    print('the accuracy under kernel %s is %f'%(kernel,clf.score(x_test,y_test)))
    print(datetime.datetime.fromtimestamp(time()-time0).strftime('%M:%S:%f'))
#为什么rbf在这个线性数据集的效果如此糟糕
#先探索数据
import pandas as pd
df=pd.DataFrame(data)
print(df.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
#  1  数据的量纲严重不统一
#  2  有些数据存在偏态问题
#尝试对数据进行标准化
from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit_transform(data)
x_train,x_test,y_train,y_test=train_test_split(x,target,test_size=0.3,random_state=420)
Kernel=['linear','poly','rbf','sigmoid或logistic函数']
for kernel in Kernel:
    time0=time()
    clf=SVC(kernel=kernel,gamma='auto',cache_size=5000)#cache_size=5000 表示允许使用多大的内存进行计算,单位mb
    clf.fit(x_train,y_train)
    print('the accuracy under kernel %s is %f'%(kernel,clf.score(x_test,y_test)))
    print(datetime.datetime.fromtimestamp(time()-time0).strftime('%M:%S:%f'))
#    标准化可以大幅度提高运行结果    大大降低运行速度

#调参
score=[]
gamma_range=np.logspace(-10,1,50)  #返回在对数刻度上的均匀间隔的数据
for i in gamma_range:
    clf=SVC(kernel='rbf',gamma=i,cache_size=5000)
    clf.fit(x_train,y_train)
    score.append(clf.score(x_test,y_test))
print(max(score),gamma_range[score.index(max(score))])#0.9766081871345029 0.012067926406393264
plt.plot(gamma_range,score)
plt.show()

#尝试对多项式核函数进行调参
#使用网格搜索
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
gamma_range=np.logspace(-10,1,20)
coef0_range=np.linspace(0,5,10)
C=np.linspace(0.01,10,20)
param_gird=dict(gamma=gamma_range,coef0=coef0_range,C=C)
cv=StratifiedShuffleSplit(n_splits=5,test_size=0.3,random_state=420)#交叉验证，数据集分5份，用0.3的数据集作为测试集
grid=GridSearchCV(SVC(kernel='rbf',degree=1,cache_size=5000),param_grid=param_gird,cv=cv)
grid.fit(x,target)
print('the best parameters are %s with a score of %0.5f'%(grid.best_params_,grid.best_score_))

#探索C
score=[]
C_range=np.linspace(0.01,30,50)
for i in C_range:
    clf=SVC(kernel='linear',C=i,cache_size=5000)
    clf.fit(x_train,y_train)
    score.append(clf.score(x_test,y_test))
print(max(score),C_range[score.index(max(score))])
plt.plot(C_range,score)
plt.show()

#换rbf
score=[]
C_range=np.linspace(0.01,30,50)
for i in C_range:
    clf=SVC(kernel='rbf',C=i,cache_size=5000)
    clf.fit(x_train,y_train)
    score.append(clf.score(x_test,y_test))
print(max(score),C_range[score.index(max(score))])
plt.plot(C_range,score)
plt.show()