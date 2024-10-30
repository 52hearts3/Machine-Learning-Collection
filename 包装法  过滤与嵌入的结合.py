#包装法的计算成本在嵌入法和过滤法之间
#包装法是最高效的特征选择方法
#feature_selection.REF
#参数  estimator  填写实例化后的评估器
#n_features_to_select  选择想要的特征的个数
#step  每次迭代删除的特征
#.suppot_  返回所有特征是否被选中的布尔矩阵
#.ranking_  返回特征的按次数迭代中综合重要性的排名
#
import pandas as pd
data=pd.read_csv(r'D:\game\sklearn\特征选择\digit recognizor.csv')
print(data.head())
x=data.iloc[:,1:]
y=data.iloc[:,0]
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rfc=RandomForestClassifier(n_estimators=50,random_state=0)
selector=RFE(rfc,n_features_to_select=360,step=50)
selector.fit(x,y)
print(selector.support_)
print(selector.ranking_)#排名越靠前，重要性越高
x_wrapper=selector.transform(x)
rfc.fit(x_wrapper,y)
score=cross_val_score(rfc,x_wrapper,y,cv=5).mean()
print(score)