import pandas as pd
data=pd.read_csv(r'D:\game\sklearn\特征选择\digit recognizor.csv')
print(data.head())
x=data.iloc[:,1:]
y=data.iloc[:,0]
print(x.shape)#维度非常高  维度指特征的数量
#用高维度数据集举例体现特征工程重要性

#1 方差过滤 VarianceThreshold
#threshold  舍弃所有方差小于threshold的特征，不填默认为0，即删除所有的记录都相同的特征
from sklearn.feature_selection import VarianceThreshold
selector=VarianceThreshold()#不填参数默认删除方差为0的特征
x_var0=selector.fit_transform(x)
print(x_var0.shape)
df=pd.DataFrame(data=x_var0)
print(df.head())

#如果想舍弃一半特征，只需填入所有特征方差的中位数
#计算中位数
import numpy as np
print(x.var())#x.var()查看x每一列的方差
me=np.median(x.var().values)
print(me)
x_fsvar=VarianceThreshold(threshold=me).fit_transform(x)
print(x_fsvar.shape)

#若特征是伯努利随机变量（0，1），假设p=0.8（0占0.8）
x_bvar=VarianceThreshold(threshold=0.8*(1-0.8)).fit_transform(x)#即二分类特征中某种分类占百分之八十以上的时候删除特征
print(x_bvar.shape)

#对于大型特征来说，对于knn算法，过滤后效果非常明显 准确率稍有提升，平均运行时间减少
#过滤前knn算法运行了半个小时  过滤后运行了20分钟
#随机森林过滤前只需11.5秒！！ 随机森林过滤后速度没快太多
#方差过滤对随机森林没有太大影响，对knn影响很大

#所以过滤法的主要对象是需要遍历特征或者需要升维的算法
#过滤法的主要目的是降低时间消耗

#选取超参数threshold
#我们怎样知道，方差过滤掉的到底时噪音还是有效特征呢？过滤后模型到底会变好还是会变坏呢？
# 答案是：每个数据集不一样，只能自己去尝试。这里的方差阈值，其实相当于是一个超参数，要选定最优的超参数，我们可以画学习曲线，找模型效果最好的点。
# 但现实中，我们往往不会这样去做，因为这样会耗费大量的时间。
# 我们只会使用阈值为0或者阈值很小的方差过滤，来为我们优先消除一些明显用不到的特征，然后我们会选择更优的特征选择方法继续削减特征数量。