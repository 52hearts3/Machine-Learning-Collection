import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
weather=pd.read_csv(r'D:\game\sklearn\支持向量机  下\weatherAUS5000.csv',index_col=0)
print(weather.head())
x=weather.iloc[:,:-1]
y=weather.iloc[:,-1]
print(x.shape)
print(x.info())
#探索缺失值
print(x.isnull().mean())#加mean()之后表示缺失值占总样本的比例
#我们要有不同的缺失值填补策略
#探索标签分类
print(np.unique(y))#['No' 'Yes']
print(y.info())#标签没有缺失

#分集，优先探索标签
#在现实中，我们会先分训练集和测试集，然后再进行数据预处理
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=420)
#恢复索引
for i in [x_train,x_test,y_train,y_test]:
    i.index=range(i.shape[0])

#查看标签是否存在样本不均衡的问题
print(y_train.value_counts())
print(y_test.value_counts())
#有轻微的样本不均衡问题
#对标签编码
from sklearn.preprocessing import LabelEncoder #标签专用
encoder=LabelEncoder().fit(y_train) #LabelEncoder允许输入一维数据
#使用训练集进行训练，然后在训练集和测试集上分别进行transform
y_train=pd.DataFrame(encoder.transform(y_train))
y_test=pd.DataFrame(encoder.transform(y_test))
#标签变为了0和1

#对特征进行描述性统计,查看有没有偏态以及异常值，存在偏态用标准化
print(x_train.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
print(x_test.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
#如果发现了异常值，首先观察异常值出现的频率
#如果异常值只出现了一次，多半是输入错误，直接把异常值删除
#如果出现了多次，与业务人员沟通
#如果异常值占到了总数的10%左右，把异常值替换成非异常但是非干扰的项，如用0进行替换，或者把异常值当缺失值处理
#在原有的15w行数据集中是有异常值存在的
#修改异常值代码
#观察异常值是否大量存在
#s1=x_train.loc[x_train.loc[:,'Cloud9am']==9,'Cloud9am']
#s2=x_test.loc[x_test.loc[:,'Cloud9am']==9,'Cloud9am']
#s3=x_test.loc[x_test.loc[:,'Cloud3am']==9,'Cloud3am']
#发现少数存在，采取删除策略
#x_train=x_train.drop(index=71737)
#y_train=y_train.drop(index=71737)
#x_test=x_test.drop(index=[19646,29632])
#y_test=y_test.drop(index=[19646,29632])
#恢复索引
#for i in [x_train,x_test,y_train.y_test]:
#    i.index=range(i.shape[0])

#处理困难特征，日期
#探索我们现在用于的日期是连续型特征，还是分类型特征
#如2024-5-1
#日期是分类365类的分类型变量
#先判断我们的日期特征中，日期是否有重复
x_train_c=x_train.copy()
print(x_train_c.iloc[:,0].value_counts())
#日期有重复，不是连续型变量
#首先，日期不是独一无二的，日期有重复
#其次，在我们分训练集和测试集之后，日期也不是连续的，而是分散的
#某年的某一天会倾向于下雨？或者倾向于不会下雨？
#不是日期影响了下雨与否，反而更多的是这一天的日照时间，湿度，温度等等这些因素影响了是否会下雨
#光看日期，其实感觉它对我们的判断并无直接影响
#如果我们把日期当作连续型变量处理，那么算法会认为它是一系列1到3000左右的数字，不会意识到这是日期
print(x_train_c.iloc[:,0].value_counts().count())#2141
#如果我们把它当作分类型变量处理，类别太多,有2141类，如果换为数值型，会被直接当作连续型变量
#从这些思考来看，既然算法处理的是列与列之间的关系，我们是否可以把‘今天的天气会影响明天的天气’这个指标转换成一个特征呢
print(x_train['Rainfall'].value_counts())#我们假设Rainfall>1就是下雨（根据描述性统计）
print(x_train['Rainfall'].isnull().sum())#33 有空值
x_train.loc[x_train['Rainfall']>=1,'RainToday']='yes'
x_train.loc[x_train['Rainfall']<1,'RainToday']='no'
x_train.loc[x_train['Rainfall']==np.nan,'RainToday']=np.nan
x_test.loc[x_test['Rainfall']>=1,'RainToday']='yes'
x_test.loc[x_test['Rainfall']<1,'RainToday']='no'
x_test.loc[x_test['Rainfall']==np.nan,'RainToday']=np.nan
print(x_train['RainToday'].value_counts())
#现在我们是否可以将日期删除了呢，对我们而言，日期本身并不影响天气，但是日期所在的月份会影响天气，如梅雨时节
#因此，我们可以将月份或者季节提取出来，作为一个特征使用，从而舍弃掉具体的日期
#int(x_train.loc[0,'Date'].split('-')[1])  提取月份
x_train['Date']=x_train['Date'].apply(lambda x:int(x.split('-')[1]))
#替换完毕后，我们需要修改列名
#rename是比较少有的，可以用来修改单个列名的函数
#通常我们都直接使用df.columns=某个列表  这样的形式来一次性修改所有的列名
#但rename允许我们只修改单独的列
x_train=x_train.rename(columns={'Date':'Month'})
print(x_train.head())
x_test['Date']=x_test['Date'].apply(lambda x:int(x.split('-')[1]))
x_test=x_test.rename(columns={'Date':'Month'})
print(x_test.head())

#处理困难特征  地点
#不同城市有着不同的下雨倾向，但尴尬的是，和时间一样，我们输入的地点的名字对于算法是一串字符，伦敦和北京对于算法来说，和0与1没有区别
#同样，我们的样本中含有49个地点，如果做成分类变量，算法就无法辨别它究竟是否是分类变量
#也就是说，我们需要让算法意识到，不同的地点因为气候不同，所以对‘明天是否会下雨’有着不同的影响
#如果我们将不同城市的地点转换为这个地方的气候的话，我们就可以将不同城市打包在同一个气候中，而在同一个气候下反应的降雨情况是相似的
print(x_train.loc[:,'Location'].value_counts().count())
#超过25个类别的分类型变量，都会被算法当作是连续型变量
cityll=pd.read_csv(r'D:\game\sklearn\支持向量机  下\cityll.csv',index_col=0)
#cityll是每个城市对应的经纬度
city_climate=pd.read_csv(r'D:\game\sklearn\支持向量机  下\Cityclimate.csv')
#city_climate是每个城市对应的气候
print(cityll.head())
print(city_climate.head())
#对经纬度处理，去掉度数符号  如34.9285°
cityll['Latitudenum']=cityll['Latitude'].apply(lambda x:float(x[:-1])) #不包括最后一个字符
cityll['Longitudenum']=cityll['Longitude'].apply(lambda x:float(x[:-1]))
#澳大利亚全部在南纬  东经
#所以经纬度的方向就舍弃了
citylld=cityll.iloc[:,[0,5,6]]
#将city_climate中的气候添加到citylld中
citylld=pd.merge(citylld,city_climate,on='City',how='inner')
print(citylld.head())
#print(citylld.loc[:,'Climate'].value_counts())
#samplecity这个csv文件是澳大利亚气象站的经纬度
sample_city=pd.read_csv(r'D:\game\sklearn\支持向量机  下\samplecity.csv',index_col=0)
#print(sample_city.head())
#对sample_city做同样的处理，舍弃经纬度的度数符号，并舍弃我们的经纬度方向
sample_city['Latitudenum']=sample_city['Latitude'].apply(lambda x:float(x[:-1]))
sample_city['Longitudenum']=sample_city['Longitude'].apply(lambda x:float(x[:-1]))
sample_cityd=sample_city.iloc[:,[0,5,6]]
print(sample_cityd.head())
#计算气象站到各个城市的距离（根据经纬度计算），依次判断气象站属于哪个城市
#使用radians将经纬度转换为弧度
from math import radians,sin,cos,acos
citylld.loc[:,'slat']=citylld.iloc[:,1].apply(lambda x:radians(x))
citylld.loc[:,'slon']=citylld.iloc[:,2].apply(lambda x:radians(x))
sample_cityd.loc[:,'elat']=sample_cityd.iloc[:,1].apply(lambda x:radians(x))
sample_cityd.loc[:,'elon']=sample_cityd.iloc[:,2].apply(lambda x:radians(x))
print(sample_cityd.head())
import sys
for i in range(sample_cityd.shape[0]):
    slat=citylld.loc[:,'slat']
    slon=citylld.loc[:,'slon']
    elat=sample_cityd.loc[i,'elat']
    elon=sample_cityd.loc[i,'elon']
    #计算每个气象站到各个城市的距离
    dist=6371.01*np.arccos(np.sin(slat)*np.sin(elat)+np.cos(slat)*np.cos(elat)*np.cos(slon.values-elon))
    city_index=np.argsort(dist)[0] #对距离进行排序
    #每次计算后，取离气象站距离最近的城市，然后将最近的城市和城市对应的气候都匹配到sample_cityd中
    sample_cityd.loc[i,'closest_city']=citylld.loc[city_index,'City']
    sample_cityd.loc[i,'climate']=citylld.loc[city_index,'Climate']
print(sample_cityd.head())
#查看气候分布
print(sample_cityd['climate'].value_counts())
#确认无误后，取出样本城市所带的气候，并保存
locafinal=sample_cityd.iloc[:,[0,-1]]
locafinal.columns=['Location','Climate']
#这里设定locafinal的索引为地点，方便之后使用map匹配
locafinal=locafinal.set_index(keys='Location')
print(locafinal.info())
#print(locafinal.loc['Katherine',:])
#开始匹配，在这里使用map功能，map能够将特征的值一一对应到我们设定的字典当中，并用字典的值来替换样本中原本的值
#将location中的内容替换，并确保匹配进入的气候字符串中不含逗号，气候两边不能有空格
#我们使用re这个模块来消除逗号
#re.sub(希望替换的值，希望被替换的值，要操作的字符串)
#x.strip() 是去掉两边空格的函数
print(x_train.head())#注意，这里的locafinal还是框架
a=pd.Series(data=locafinal.iloc[:,0],index=locafinal.index)
print(a.head())
dict=a.to_dict()
import re
#map函数既可以输入字典，还可以输入框架
x_train['Location']=x_train['Location'].map(dict).apply(lambda x:re.sub(',','',x.strip()))
x_test['Location']=x_test['Location'].map(locafinal.iloc[:,0]).apply(lambda x:re.sub(',','',x.strip()))
#print(x_train.head())
#修改列名
x_train=x_train.rename(columns={'Location':'Climate'})
x_test=x_test.rename(columns={'Location':'Climate'})
print(x_train.head())
print(x_test.head())
x_train.to_csv(r'D:\game\sklearn\支持向量机  下\x_train.csv')
x_test.to_csv(r'D:\game\sklearn\支持向量机  下\x_test.csv')
y_train.to_csv(r'D:\game\sklearn\支持向量机  下\y_train.csv')
y_test.to_csv(r'D:\game\sklearn\支持向量机  下\y_test.csv')