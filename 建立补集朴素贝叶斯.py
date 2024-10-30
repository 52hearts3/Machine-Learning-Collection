import numpy as np
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB,ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import brier_score_loss,recall_score,roc_auc_score
class_1=50000#多数类样本数目
class_2=500#少数类样本数目
centers=[[0.0,0.0],[5.0,5.0]]#设定两个类别的中心
clusters_std=[3,1]#设置两个类的方差
x,y=make_blobs(n_samples=[class_1,class_2],centers=centers,cluster_std=clusters_std,random_state=0,shuffle=False)
print(x.shape)
print(np.unique(y))
name=['Multinomial','Gaussian','Bernoulli','Complement']
models=[MultinomialNB(),GaussianNB(),BernoulliNB(),ComplementNB()]
for name,clf in zip(name,models):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=420)
    if name!='Gaussian':#干脆全部做成二分类 0，1
        kbs=KBinsDiscretizer(n_bins=10,encode='onehot')#有两个特征，每个特征分10类，就变为了20个特征
        kbs.fit(x_train)
        x_train=kbs.transform(x_train)
        x_test=kbs.transform(x_test)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    proba=clf.predict_proba(x_test)[:,1]
    score=clf.score(x_test,y_test)
    print(name)
    print('  Brier:{:.3f}'.format(brier_score_loss(y_test,proba,pos_label=1)))
    print('  accuracy:{:.3f}'.format(score))
    print('  recall:{:.3f}'.format(recall_score(y_test,y_pred)))
    print('  auc:{:.3f}'.format(roc_auc_score(y_test,proba)))
#根据结果知道伯努利贝叶斯对少数类的捕捉能力更强，但是是在自建的数据集上强行二值化得到的结果
#因此我们有了改进的多项式贝叶斯   补集朴素贝叶斯