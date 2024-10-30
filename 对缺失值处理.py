import pandas as pd
data=pd.read_csv('Narrativedata.csv',index_col=0)#index_col=0 把第0列当成索引
print(data)
print(data.info())
#使用SimpleImputer 填补缺失值
#参数  missing_values  告诉SimpleImputer，数据中的缺失值长什么样，默认为np.nan
#strategy 填补缺失值的策略 默认均值  输入mean均值（仅对数值型数据可用） 输入median中值（仅对数值型数据可以） 输入most_frequent众数（对数值型和字符型都可用）
#输入constant表示请参考参数fill_value中的值（对字符型和数字型都可用）
#fill_value 当strategy为constant时可用  可输入字符或数字表示要填充的值，常用0
#copy 默认为True 将创建特征矩阵的副本，反之会将缺失值填补到原本的特征矩阵中

#填补年龄
age=data.loc[:,'Age'].values.reshape(-1,1)#sklearn中的特征矩阵必须是二维的
#Series必须用values转为array才能用reshape
from sklearn.impute import SimpleImputer
imp_mean=SimpleImputer()#默认为均值
imp_media=SimpleImputer(strategy='median')#中位数
imp_0=SimpleImputer(strategy='constant',fill_value=0)#0
imp_mean=imp_mean.fit_transform(age)
imp_media=imp_media.fit_transform(age)
imp_0=imp_0.fit_transform(age)

#这里使用中位数填补age
data.loc[:,'Age']=imp_media
print(data.info())
#填Embarked
Embarked=data.loc[:,'Embarked'].values.reshape(-1,1)#Series必须用values转为array才能用reshape
print(Embarked)
imp_mode=SimpleImputer(strategy='most_frequent')
data.loc[:,'Embarked']=imp_mode.fit_transform(Embarked)
print(data.info())

#用pandas处理缺失值
data=pd.read_csv('Narrativedata.csv',index_col=0)
data.loc[:,'Age']=data.loc[:,'Age'].fillna(data.loc[:,'Age'].median())
data.dropna(axis=0,inplace=True)
print(data.info())