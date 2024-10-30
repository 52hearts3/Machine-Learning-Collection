#feature_selection.f_classif   f检验分类
#feature_selection.f_regression  f检验回归
#当数据服从正态分布时效果非常稳定，如果使用我们会先把数据转为服从正态分布的方式（标准化）
#F检验的本质是寻找两组数据之间的线性关系，和卡方过滤一样，我们希望选取p<0.05或0.01的特征，这些特征与标签有显著的线性关系
#当p>0.05或0.01是，这些特征被我们认为是和标签没有线性关系的，应该被删除
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
data=pd.read_csv(r'D:\game\sklearn\特征选择\digit recognizor.csv')
print(data.head())
x=data.iloc[:,1:]
y=data.iloc[:,0]
x_new=VarianceThreshold(threshold=0).fit_transform(x)#先方差过滤
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
f,p=f_classif(x_new,y)
print(f)
print(p)
k=f.shape[0]-(p>0.05).sum()     #重要！！！！！！！
print(k)
x_fsF=SelectKBest(f_classif,k=k).fit_transform(x_new,y)  #SelectKBest需要x和y
rfc=RandomForestClassifier(n_estimators=20,random_state=0)
rfc.fit(x_fsF,y)
score=cross_val_score(rfc,x_fsF,y,cv=5)
print(score)

#f检验只能检验出线性关系
#互信息法可以找出所有关系
#比f检验更强大
#在feature_selection模块中的mutual_info_classif与mutual_info_regression
#互信息法不返回p值或f值这种类似的统计量。它返回0，1 0表示两个变量独立，为1表示两个变量相关  #在0到1之间就有关系
from sklearn.feature_selection import mutual_info_classif
result=mutual_info_classif(x_new,y)
print(result)
#找出大于0的索引
import numpy as np
x_selected=np.where(result>0)[0]#np.where返回的是一个索引数组，因为result是一维数组，np.where(result>0)[0]表示我们只要它的索引
print(x_selected)#因为result是一维数组,这里np.where(result>0)[0]返回的是列索引
x_1=x_new[:,x_selected]
print(x_1)
#检验
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rfc=RandomForestClassifier(n_estimators=20,random_state=0)
rfc.fit(x_1,y)
score=cross_val_score(rfc,x_1,y,cv=5).mean()
print(score)