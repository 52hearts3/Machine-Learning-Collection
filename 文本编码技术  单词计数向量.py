#单词计数向量
#这种计数中样本可以包含一段话或一篇文章，这个样本中如果出现了10个单词，就会有10个特征
#每个特征xi代表一个单词，特征xi的取值代表这个单词在这个样本中总共出现了几次，是一个离散的，代表次数的正整数
#通过feature.extraction.text模块中的CounVectorizer来实现
sample=['machine learning is fascinating,it is wonderful',
        'machine learning is a sensational techonology',
        'elsa is a popular character']
from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer()
x=vec.fit_transform(sample)
print(x)
#使用get_feature_names_out调用每个列的名称
print(vec.get_feature_names_out())
#使用pandas查看
#注意稀疏矩阵是无法输入pandas的
import pandas as pd
df=pd.DataFrame(x.toarray(),columns=vec.get_feature_names_out())
print(df)#有三句话，所以有三行
#单词计数向量会出现问题
#句子越长的样本对其影响越大，我们希望尽量避免这个影响
#为了避免句子太长的样本对我们的参数估计造成太大的影响，因此补集朴素贝叶斯让每个特征的权重除以自己的L2范式，就是为了避免这种情况的发生
#另外，如果使用单词计数向量，可能会导致一部分常用词（比如中文的'的'）频繁出现在我们的矩阵中并且占有很高的权重，对于分类来说，这明显是对算法的一种误导
#为了避免这个问题，我们使用著名的TF-IDF方法