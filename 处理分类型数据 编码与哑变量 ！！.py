#一些数据收集完毕后，都不是以数字来表现的，这种情况下，我们必须将数据进行编码，把文字型数据转换为数值型
import pandas as pd
data=pd.read_csv(r'D:\game\sklearn\数据预处理\Narrativedata.csv',index_col=0)
print(data)
#LabelEncoder 标签专用，将分类转换为分类数值       标签专用!!!!!!!!!!!
from sklearn.preprocessing import LabelEncoder
y=data.iloc[:,-1]#因为这里输入的是标签，不是特征矩阵，所以允许一维
le=LabelEncoder()#实例化
le.fit(y)
label=le.transform(y)
print(label)#0,1,2
print(le.classes_)#查看类别  重要！！！！
print(le.inverse_transform(label))#逆转
data.iloc[:,-1]=label
print(data.head())

#OrdinalEncoder 特征专用，能将分类特征转换为分类数值      特征专用!!!!!
#因为是特征专用，所以不能导入一维数组
from sklearn.preprocessing import OrdinalEncoder
data_=data.copy()
print(data_.head())
s=OrdinalEncoder().fit(data_.iloc[:,1:3]).categories_#data_.iloc[:,1:3]指取索引为1到2列，不包括索引为3的列
#data_.iloc[:,1:-1]指从索引为1的列取到最后一列（不包括最后一列）
print(s)#categories_查看每个特征中有多少个类别 ！！！
data_.iloc[:,1:-1]=OrdinalEncoder().fit_transform(data_.iloc[:,1:-1])
print(data_.head())#Sex Embarked都变为了数字

#我们使用0，1，2代表了三个不同的舱门，这样转换是正确的吗？
#思考三种不同性质的分类数据
# 1 舱门 s c q
#三种取值s c q是彼此独立的，彼此之间完全没有联系，表达的是s！=q！=c的概念，这是名义变量
# 2  学历 小学 初中 高中
#三种取值不是完全独立的，在性质上有高中>初中>小学这样的联系 学历有高有低，但是学历之间却是不可以计算的，我们不能说小学+某个取值=初中，这是有序变量
# 3  体重 45kg 90kg 135kg
#各个取值之间有联系，且是可以互相计算的，分类之间可以通过数学计算相互转换，这是有距变量

#然而在对特征进行编码的时候，这三种分类数据都会被我们转换为［0,1,2]，这三个数字在算法看来，是连续且可以计算的，
# 这三个数字相互不等，有大小，并且有着可以相加相乘的联系。所以算法会把舱门，学历这样的分类特征，都误会成是体重这样的分类特征。
# 这是说，我们把分类转换成数字的时候，忽略了数字中自带的数学性质，所以给算法传达了一些不准确的信息，而这会影响我们的建模

#对于性别 舱门这种内部之间没有联系的特征（名义变量），我们用哑变量最准确
#独热编码 OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
x=data.iloc[:,1:-1]
print(x.head())
enc=OneHotEncoder(categories='auto')
enc.fit(x)
result=enc.transform(x).toarray()#变为array格式    本身为元组格式，toarray非常重要！！！！！！！！
print(result)
#还原
print(pd.DataFrame(enc.inverse_transform(result)))
#重要属性  get_feature_names_out()  告诉我们哪个哑变量属于哪个特征
print(enc.get_feature_names_out())
#把矩阵放到原数据集中
new_data=pd.concat([data,pd.DataFrame(result)],axis=1)
print(new_data.head())
new_data.drop(['Sex','Embarked'],axis=1,inplace=True)
new_data.columns=['Age','Survived','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S','Embarked_nan']
print(new_data.head())