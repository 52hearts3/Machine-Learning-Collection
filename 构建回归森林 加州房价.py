from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
fetch_california_housing=fetch_california_housing()
data=fetch_california_housing.data
target=fetch_california_housing.target
feature_names=fetch_california_housing.feature_names
df=pd.DataFrame(data=data,columns=feature_names)
print(df)
#实例化
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
score=cross_val_score(regressor,data,target,cv=10,scoring='neg_mean_squared_error')
print(score)
import sklearn
sorted(sklearn.metrics.SCORERS.keys())#列出sklearn中模型的评分指标