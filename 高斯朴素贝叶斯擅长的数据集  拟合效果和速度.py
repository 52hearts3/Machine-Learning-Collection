#高斯朴素贝叶斯属于比较特殊的一类分类器
#其分类效果在二分类数据和月亮型数据上表现优秀，但是在环形数据上不太擅长
#之前我们学过的其他模型，许多线性模型如逻辑回归，线性SVM等，在线性数据集上会绘制直线决策边界，因此对月亮型和环形数据难以区分
#但是高斯朴素贝叶斯的决策边界是曲线，尽管它更擅长可分得二分类数据，但朴素贝叶斯在环形数据和月亮型数据上也有远胜过其他模型的表现

#但是这些的前提是特征之间是独立的
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve #画学习曲线的类
from sklearn.model_selection import ShuffleSplit #定义交叉验证模式的类
from time import time
import datetime
#定义画学习曲线函数
def plot_learning_curve(estimator,title,x,y,ax,ylim=None,cv=None,n_jobs=None): #ax为选择子图,estimator为分类器
    train_sizes,train_scores,test_scores=learning_curve(estimator,x,y,cv=cv,n_jobs=n_jobs)#详细见疑难困惑
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)#为了保证五张图的纵坐标一致
    ax.set_xlabel('training examples')
    ax.set_ylabel('score')
    ax.grid()#显示网格作为背景
    ax.plot(train_sizes,np.mean(train_scores,axis=1),'o-',color='r',label='training score')#o-表示带点的直线
    ax.plot(train_sizes,np.mean(test_scores,axis=1),'o-',color='g',label='test score')
    ax.legend(loc='best')
    return ax
digits=load_digits()
x,y=digits.data,digits.target
title=['naive bytes','decisiontree','SVM RBF kernel','RandomForest','Logistic']
model=[GaussianNB(),DecisionTreeClassifier(),SVC(gamma=0.001),RandomForestClassifier(n_estimators=50),LogisticRegression(C=0.1,solver='lbfgs',max_iter=1000)]
cv=ShuffleSplit(n_splits=50,test_size=0.2,random_state=0)
fig,axes=plt.subplots(1,5,figsize=(30,6))#生成一行五列画布
for ind,title_,estimator in zip(range(len(title)),title,model):
    times=time()
    plot_learning_curve(estimator,title_,x,y,ax=axes[ind],ylim=[0.7,1.05],n_jobs=4,cv=cv)
    print('{}:{}'.format(title_,datetime.datetime.fromtimestamp(time()-times).strftime('%M:%S:%f')))
plt.show()

#贝叶斯天生不如其他分类器强大
#当训练数据集越大，贝叶斯的训练准确率越差

#贝叶斯很容易接近极限
#如果我们追求概率预测，首选逻辑回归，其次是贝叶斯