#ROC曲线是以不同阈值下的假正率FDR为横坐标，不同阈值下的召回率（recall）为纵坐标的曲线
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
class_1=7
class_2=4
centers=[[0.0,0.0],[1,1]]
clusters_std=[0.5,1]
x,y=make_blobs(n_samples=[class_1,class_2],centers=centers,cluster_std=clusters_std,random_state=0,shuffle=False)
plt.scatter(x[:,0],x[:,1],c=y,cmap='rainbow',s=30)
plt.show()
#使用逻辑回归建模
clf_lo=LogisticRegression()
clf_lo.fit(x,y)
prob=clf_lo.predict_proba(x) #返回的是每个样本预测为0，1（y的值）的概率
#print(prob)
#将样本和概率放在一个框架中
import pandas as pd
prob=pd.DataFrame(prob)
prob.columns=['0','1']
#print(prob)
#手动调节阈值，来改变我们的模型效果
for i in range(prob.shape[0]): #对框架的每行进行循环
    if prob.loc[i,'1']>0.5:
        prob.loc[i,'pred']=1 #如果不存在pred列则创造一列
    else:
        prob.loc[i,'pred']=0
print(prob)
#把真实的标签添加进去
prob['y_true']=y
#对'1'这一列排序
prob=prob.sort_values(by='1',ascending=False)#ascending=False不要逆序
from sklearn.metrics import confusion_matrix,recall_score
#获取混淆矩阵
cm=confusion_matrix(prob.loc[:,'y_true'],prob.loc[:,'pred'],labels=[1,0])
#第一个参数为真实值，第二个参数为预测值，还有labels[a,b] a是少数类，b是多数类
print(cm)
#假正率=1-特异度
#特异度=完全预测正确的多数类/所有真实的多数类
#故 假正率=预测错误的多数类/所有真实的多数类
FPR=cm[1,0]/cm[1,:].sum()
#recall=所有预测为少数类的点/全部为少数类的点
recall=cm[0,0]/cm[0,:].sum()

#绘制ROC曲线
#概率 clf.proba.predict_proba(x)[:,1] 代表类别1下的概率
#阈值 每一个阈值都代表着一次循环，每一次循环，都会有一个混淆矩阵，要有一组假正率和recall
#np.linspace(概率最小值，概率最大值，55,endpoint=False) #取百分之10的数据画图 endpoint=False表示不要取到最大值
#开始绘图
recall=[]
FPR=[]
probrange=np.linspace(clf_lo.predict_proba(x)[:,1].min(),clf_lo.predict_proba(x)[:,1].max(),55,endpoint=False)#阈值
for i in probrange:
    y_predict=[]
    for j in range(x.shape[0]):
        if clf_lo.predict_proba(x)[j,1]>i: #判断x中每个样本预测概率是否大于阈值
            y_predict.append(1)
        else:
            y_predict.append(0)
    cm=confusion_matrix(y,y_predict,labels=[1,0])
    recall.append(cm[0,0]/(cm[0,:].sum()))
    FPR.append(cm[1,0]/(cm[1,:].sum()))
recall.sort()
FPR.sort()
plt.plot(FPR,recall,c='red')
plt.plot([0,1],[0,1],c='black',linestyle='--')
plt.show()
#对于一条凸性的ROC曲线来说，曲线越靠近左上角越好，越往下越糟糕，如果曲线在虚线下方，则证明模型完全无法使用
#对于一条凹形的ROC曲线来说，应该越靠近右下角越好，凹形曲线代表模型的预测结果与真实情况完全相反，那也不算糟糕，我们手动将模型逆转即可得到凸型曲线（调整混淆矩阵的labels）
#最糟糕的是无论曲线是凸性还是凹形，曲线位于图像中间，和虚线非常接近，那我们拿他无能为力