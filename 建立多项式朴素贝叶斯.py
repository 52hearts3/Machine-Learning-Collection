import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss
class_1=500
class_2=500#两个类别分别设定500个样本
centers=[[0.0,0.0],[2.0,2.0]]#设定两个类别的中心
clusters_std=[0.5,0.5]#设定两个类别的方差
x,y=make_blobs(n_samples=[class_1,class_2],centers=centers,cluster_std=clusters_std,random_state=0,shuffle=False)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=420)

#归一化，防止训练集和测试集出现负数
mms=MinMaxScaler()
mms.fit(x_train)
x_train=mms.transform(x_train)
x_test=mms.transform(x_test)

#建模
mnb=MultinomialNB()
mnb.fit(x_train,y_train)

#重要属性 根据数据获得的，每个标签类的对数先验概率log(P(y))
#由于概率永远在0到1之间，因此对数先验概率返回的永远是负数
print(mnb.class_log_prior_)#永远等于标签中所带的类别数量
#可以使用np.exp查看真正的概率
print(np.exp(mnb.class_log_prior_))

#重要属性，返回一个固定标签类别下的每个特征的对数概率
print(mnb.feature_log_prob_)#有两个特征，两个标签，返回的行数代表标签，列数代表特征
print(np.exp(mnb.feature_log_prob_))

#重要属性  在fit时每个标签类别下包含的样本数
#当fit接口中的sample_weight被设置时，该接口返回的值也会受到加权的影响
print(mnb.class_count_)#[351. 349.]  0有351个，1有349个

#一些传统接口
print(mnb.predict(x_test))
print(mnb.predict_proba(x_test))#每个样本在每个标签下的取值
print(mnb.score(x_test,y_test))
print(brier_score_loss(y_test,mnb.predict_proba(x_test)[:,1],pos_label=1))

#模型效果不好，因为多项式朴素贝叶斯擅长分类型数据
#试试把训练集转换为分类型数据，对连续型变量进行分箱
from sklearn.preprocessing import KBinsDiscretizer
kbs=KBinsDiscretizer(n_bins=10,encode='onehot')
kbs.fit(x_train)
x_train_=kbs.transform(x_train)
x_test_=kbs.transform(x_test)
#变为20个特征，是因为有2个特征，每个特征分了10个箱所分出来的哑变量
mnb=MultinomialNB()
mnb.fit(x_train_,y_train)
print(mnb.score(x_test_,y_test))
print(brier_score_loss(y_test,mnb.predict_proba(x_test_)[:,1],pos_label=1))

#由此可知，只要对多项式朴素贝叶斯做分类处理，效果一定会突飞猛进，作为在文本分类中大放异彩的算法