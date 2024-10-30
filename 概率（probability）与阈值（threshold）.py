#要理解概率与阈值，最容易的状况是来回忆一下我们用逻辑回归做分类的时候的状况。
# 逻辑回归的predict_proba 接口对每个样本生成每个标签类别下的似然（类概率）。
# 对于这些似然，逻辑回归天然规定，当一个样本所对应的这个标签类别下的似然大于0.5的时候，这个样本就被分为这一类。
# 比如说，一个样本在标签1下的似然是0.6，在标签0下的似然是0.4，则这个样本的标签自然就被分为1。
# 逻辑回归的回归值本身，其实也就是标签1下的似然。在这个过程中，0.5就被称为阈值。
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
print(prob)
#将样本和概率放在一个框架中
import pandas as pd
prob=pd.DataFrame(prob)
prob.columns=['0','1']
print(prob)
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
print(prob)

#使用混淆矩阵
from sklearn.metrics import confusion_matrix,precision_score,recall_score
s=confusion_matrix(prob.loc[:,'y_true'],prob.loc[:,'pred'],labels=[1,0])#第一个参数为真实值，第二个参数为预测值，还有labels[a,b] a是少数类，b是多数类
# 混淆矩阵还有sample_weight参数
print(s)#主对角线是判断正确的，右对角线是判断错误的
#试试看利用混淆矩阵手动计算precision和recall
#precision=所有判断正确并确实为1的样本/所有被判为1的样本   （在此案例中，少数类的y为1）
#recall=所有预测为1的点/全部为1的点  （在此案例中，少数类的y为1）

#使用sklearn计算precision和recall
precision=precision_score(prob.loc[:,'y_true'],prob.loc[:,'pred'],labels=[1,0])
#第一个参数为真实值，第二个参数为预测值，还有labels[a,b] a是少数类，b是多数类
print(precision)
recall=recall_score(prob.loc[:,'y_true'],prob.loc[:,'pred'],labels=[1,0])
#第一个参数为真实值，第二个参数为预测值，还有labels[a,b] a是少数类，b是多数类
print(recall)

#加入我们使用0.4为阈值呢
for i in range(prob.shape[0]): #对框架的每行进行循环
    if prob.loc[i,'1']>0.4:
        prob.loc[i,'pred']=1 #如果不存在pred列则创造一列
    else:
        prob.loc[i,'pred']=0
print(prob)
#使用sklearn计算precision和recall
precision=precision_score(prob.loc[:,'y_true'],prob.loc[:,'pred'],labels=[1,0])
#第一个参数为真实值，第二个参数为预测值，还有labels[a,b] a是少数类，b是多数类
print(precision)
#precissiom反应的是在我们预测的少数类样本中，真正的少数类所占的比例
recall=recall_score(prob.loc[:,'y_true'],prob.loc[:,'pred'],labels=[1,0])
#第一个参数为真实值，第二个参数为预测值，还有labels[a,b] a是少数类，b是多数类
#recall反应的是模型捕获少数类的能力
print(recall)
#也就是说，我们可以不断地调整一个阈值，来得到我们想要的最好的精确度或召回率，假正率
#注意，降低或增高阈值并不一定能让模型的效果更好，一切都基于我们想要追求怎样的模型效果
#通常来说，降低阈值能够升高racall（召回率）

#如果我们有概率需求，我们还是会追求逻辑回归或者朴素贝叶斯，不过，SVC也可以生成概率，详细请见SVM做概率预测