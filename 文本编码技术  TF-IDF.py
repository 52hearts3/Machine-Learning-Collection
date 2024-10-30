#TF-IDF全称 词频逆文档频率，是通过单词在文档中出现的频率来衡量其权重
#也就是说，IDF的大小与一个词的常见程度成反比，这个词越常见，编码后为它设置的权重越小，以此来压制频繁出现的一些无意义的词
#使用feature_extraction.text类中的TfidVectorizer来调用
sample=['machine learning is fascinating,it is wonderful',
        'machine learning is a sensational techonology',
        'elsa is a popular character']
from sklearn.feature_extraction.text import TfidfVectorizer
vec=TfidfVectorizer()
x=vec.fit_transform(sample)#每一个单词作为一个特征，每个单词在这个句子中所占的比例
#使用接口get_feature_names_out来调用每个列的名称
import pandas as pd
df=pd.DataFrame(x.toarray(),columns=vec.get_feature_names_out())
print(df)
#可见IDF将原本出现次数较多的词进行压缩，以实现压缩我们的权重
#将原本出现次数比较少的词进行一个拓展，以实现增加我们的权重