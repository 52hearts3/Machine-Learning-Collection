from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
wine=load_wine()
data=wine.data
target=wine.target
feature_names=wine.feature_names
df=pd.DataFrame(data=data,columns=feature_names)
print(df)
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.3)
clf=DecisionTreeClassifier(random_state=0)
rfc=RandomForestClassifier(random_state=0)
clf.fit(x_train,y_train)
rfc.fit(x_train,y_train)
score_c=clf.score(x_test,y_test)
score_r=rfc.score(x_test,y_test)
print('single tree:{}'.format(score_c),'random tree:{}'.format(score_r))
#format()是把括号里的东西转到前面{}中
#交叉验证效果
s1=cross_val_score(clf,data,target,cv=10)#在交叉验证中输入完整的x，y
s2=cross_val_score(rfc,data,target,cv=10)
plt.plot(range(1,11),s1,label='DecisionTreeClassifier')#range(1,11)是交叉验证次数
plt.plot(range(1,11),s2,label='RandomForestClassifier')
plt.xticks(range(1,11))
plt.legend()
plt.show()
rfc_1=[]
clf_1=[]
#一百次交叉验证效果
for i in range(10):
     rfc=RandomForestClassifier()
     rfc_s=cross_val_score(rfc,data,target,cv=10).mean()
     clf=DecisionTreeClassifier()
     clf_s=cross_val_score(clf,data,target,cv=10).mean()
     rfc_1.append(rfc_s)
     clf_1.append(clf_s)
plt.plot(range(1,11),rfc_1,label='RandomForestClassifier')
plt.plot(range(1,11),clf_1,label='DecisionTreeClassifier')
plt.legend()
plt.show()
#随机森林n_estimators学习曲线
superpa=[]
for i in range(200):
     rfc=RandomForestClassifier(n_estimators=i+1)
     rfc_s=cross_val_score(rfc,data,target,cv=10).mean()
     superpa.append(rfc_s)
print(max(superpa),superpa.index(max(superpa))+1)
plt.figure(figsize=(20,5))
plt.plot(range(1,201),superpa)
plt.show()