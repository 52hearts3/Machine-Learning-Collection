#卡方过滤是专门针对分类问题的相关性过滤
#卡方检验每个非负特征与标签之间的卡方统计量，并依照卡方统计量由高到低为特征排名
#如果卡方检测到某个特征的所有值都相同，会提示我们先使用方差进行方差过滤
import pandas as pd
data=pd.read_csv(r'D:\game\sklearn\特征选择\digit recognizor.csv')
print(data.head())
x=data.iloc[:,1:]
y=data.iloc[:,0]
from sklearn.feature_selection import VarianceThreshold
fe=VarianceThreshold(threshold=0)
x_new=fe.fit_transform(x)
df=pd.DataFrame(x_new)
print(df.head())
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest    #选择k个分数最高的类
from sklearn.feature_selection import chi2      #卡方检验
x_fschi=SelectKBest(chi2,k=705).fit_transform(x_new,y)#选前300个卡方值特征
print(x_fschi.shape)
#检验效果
rfc=RandomForestClassifier(n_estimators=30,random_state=0)
rfc.fit(x_fschi,y)
score=cross_val_score(rfc,x_fschi,y,cv=10).mean()
print(score)

#如何知道k选多少，是过滤掉了模型的噪音，这个时候的过滤是有效的
#画学习曲线
#import matplotlib.pyplot as plt
#score=[]
#for i in range(390,200,-10):
    #x_fschi=SelectKBest(chi2,k=i).fit_transform(x_new,y)
    #once=cross_val_score(rfc,x_fschi,y,cv=5).mean()
    #score.append(once)
#plt.plot(range(390,200,-10),score)
#plt.show()

#以上运行时间非常长
#更好的办法  按p值选取特征  p为卡方值
#当p<=0.05或0.01 两组数据是相关的    >0.05或0.01 两组数据是独立的
#我们可以直接从chi2实例化后的模型中获得各个特征的对应的卡方值和p值
chivalue,pvalues_chi=chi2(x_new,y)#第一个是卡方值，第二个是p值         重要!!!!!!!!!!!!!!
print(chivalue)
print(pvalues_chi)
k=chivalue.shape[0]-(pvalues_chi>0.05).sum()
print(k)#k就取708