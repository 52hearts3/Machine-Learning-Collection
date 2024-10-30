import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score,recall_score
from time import time
import datetime
x_train=pd.read_csv(r'D:\game\sklearn\支持向量机  下\x_train_final.csv',index_col=0)
x_test=pd.read_csv(r'D:\game\sklearn\支持向量机  下\x_test_final.csv',index_col=0)
y_train=pd.read_csv(r'D:\game\sklearn\支持向量机  下\y_train.csv',index_col=0)
y_test=pd.read_csv(r'D:\game\sklearn\支持向量机  下\y_test.csv',index_col=0)
print(type(y_test))#这里的y是框架，需要转换结构，要么使用ravel降维，要么转换为series
y_train=pd.Series(data=y_train.iloc[:,0])
y_test=pd.Series(data=y_test.iloc[:,0])
print(y_test.shape)
times=time()
for kernel in ['linear','poly','rbf','sigmoid或logistic函数']:
    clf=SVC(kernel=kernel,gamma='auto',degree=1,cache_size=5000)
    clf.fit(x_train,y_train)
    result=clf.predict(x_test)
    score=clf.score(x_test,y_test)
    recall=recall_score(y_test,result)
    auc=roc_auc_score(y_test,clf.decision_function(x_test))#详细参数见sklearn中的ROC曲线和AUC面积
    print('%s testing accuracy %f,recall is %f,auc is %f'%(kernel,score,recall,auc))
    print(datetime.datetime.fromtimestamp(time()-times).strftime('%M:%S:%f'))
#linear的分数最高，因为数据已经标准化，所以数据就是线性关系的，我们可以放弃rbf和sigmoid了

#调参
# 1 我希望不计代价的预测下雨天（少数类），得到最高的recall
# 2 我希望追求最高的预测准确率，一切都是让accuracy最高，不必在意recall和AUC
# 3 我希望达到recall，ROC,和accuracy之间的平衡，不准求任何一个也不牺牲任何一个

# 1 求最高的recall
print('*'*60)
print('class_weight=balanced时的recall')
#我们可以调整参数class_weight参数，首先使用balanced来调节
clf=SVC(kernel='linear',gamma='auto',degree=1,cache_size=5000,class_weight='balanced')
clf.fit(x_train,y_train)
result=clf.predict(x_test)
score=clf.score(x_test,y_test)
recall=recall_score(y_test,result)
auc=roc_auc_score(y_test,clf.decision_function(x_test))#详细参数见sklearn中的ROC曲线和AUC面积
print('%s testing accuracy %f,recall is %f,auc is %f'%('linear',score,recall,auc))
print(datetime.datetime.fromtimestamp(time()-times).strftime('%M:%S:%f'))
#为了使recall更高，我们直接调权重
print('*'*60)
print('权重为1:10的recall')
clf=SVC(kernel='linear',gamma='auto',degree=1,cache_size=5000,class_weight={1:10})#1指的是y里的1，指的是少数类的权重为10
clf.fit(x_train,y_train)
result=clf.predict(x_test)
score=clf.score(x_test,y_test)
recall=recall_score(y_test,result)
auc=roc_auc_score(y_test,clf.decision_function(x_test))#详细参数见sklearn中的ROC曲线和AUC面积
print('%s testing accuracy %f,recall is %f,auc is %f'%('linear',score,recall,auc))
print(datetime.datetime.fromtimestamp(time()-times).strftime('%M:%S:%f'))

# 2 追求最高的准确率
#首先查看样本不均衡问题
value_c=y_test.value_counts()
print(value_c)
print(value_c[0]/value_c.sum())#0.7713333333333333
#结果说明如果我们的模型把结果全部判断为多数类，结果也只有77%，而原本的模型结果为84%
#说明我们在判断多数类上其实没有太大文章可做了
#初步判断，可以认为我们已经将大部分的多数类判断为正确了，所以才能够得到现在的正确率，为了更加准确的判断，我们使用混淆矩阵计算特异度
#如果特异度非常高，证明多数类已经很难被操作
from sklearn.metrics import confusion_matrix
clf=SVC(kernel='linear',gamma='auto',cache_size=5000)
clf.fit(x_train,y_train)
result=clf.predict(x_test)
cm=confusion_matrix(y_test,result,labels=(1,0))#少数类标签是1
#特异度  所有被正确预测为0的样本/所有0的样本
specificity=cm[1,1]/cm[1,:].sum()
print(specificity)#0.9550561797752809  说明模型对多数类捕获能力非常强，我们已经没有办法在准确度上做文章了
#现在只有一条路可走，就是试试把class_weight向少数类稍微调整
print('*'*60)
print('调整class_weight后精确度的变化')
i_range=np.linspace(0.01,0.05,10)
for i in i_range:
    clf = SVC(kernel='linear', gamma='auto',cache_size=5000,class_weight={1:1+i})
    clf.fit(x_train, y_train)
    result = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    recall = recall_score(y_test, result)
    auc = roc_auc_score(y_test, clf.decision_function(x_test))  # 详细参数见sklearn中的ROC曲线和AUC面积
    print('%s testing accuracy %f,recall is %f,auc is %f' % (1+i, score, recall, auc))
#这种情况调节参数已经没有办法使准确率达到质变了，我们如果想要更高，必须考虑更换模型
#尝试逻辑回归  因为是线性模型
print('*'*60)
print('尝试逻辑回归')
from sklearn.linear_model import LogisticRegression
lo=LogisticRegression(solver='liblinear')
lo.fit(x_train,y_train)
print(lo.score(x_test,y_test))
C_range=np.linspace(3,5,10)
for c in C_range:
    lo=LogisticRegression(solver='liblinear',C=c).fit(x_train,y_train)
    print(c,lo.score(x_test,y_test))

# 3 追求平衡
times=time()
clf=SVC(kernel='linear',C=3.166,cache_size=5000,class_weight='balanced')
clf.fit(x_train,y_train)
result=clf.predict(x_test)
score=clf.score(x_test,y_test)
recall=recall_score(y_test,result)
auc=roc_auc_score(y_test,clf.decision_function(x_test))
print('testing accuracy %f,recall is %f,auc is %f'%(score,recall,auc))
print(datetime.datetime.fromtimestamp(time()-times).strftime('%M:%S:%f'))
#可见更改C值提升不大
#尝试画ROC曲线
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
FPR,recall,threshold=roc_curve(y_test,clf.decision_function(x_test),pos_label=1)#正样本（少数类）标签为1
area=roc_auc_score(y_test,clf.decision_function(x_test))
plt.figure()
plt.plot(FPR,recall,color='red',label='ROC curve (area=%0.2f)'%area)
plt.plot([0,1],[0,1],color='black',linestyle='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('FPR')
plt.ylabel('recall')
plt.title('receiver operating characteristic,example')
plt.legend(loc='lower right')
plt.show()
#找最佳阈值
max_index=(recall-FPR).tolist().index(max(recall-FPR))
print(threshold[max_index])
#手动根据阈值选择预测结果
clf=SVC(kernel='linear',C=3.166,cache_size=5000,class_weight='balanced')
clf.fit(x_train,y_train)
prob=pd.DataFrame(clf.decision_function(x_test))
prob.loc[prob.iloc[:,0]>=threshold[max_index],'y_pred']=1
prob.loc[prob.iloc[:,0]<threshold[max_index],'y_pred']=0
print(prob.loc[:,'y_pred'].isnull().sum())
#检查模型本身的准确度
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,prob.loc[:,'y_pred'].values)#前边是真实值，后边是预测值
recall=recall_score(y_test,prob.loc[:,'y_pred'].values)
print('testing accuracy %f,recall is %f'%(score,recall))