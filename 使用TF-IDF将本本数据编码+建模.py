from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
categories=['sci.space',#科学技术 太空
            'rec.sport.hockey',#运动 曲棍球
            'talk.politics.guns',#政治 枪支问题
            'talk.politics.mideast']#政治 中东问题
train=fetch_20newsgroups(subset='train',categories=categories)
test=fetch_20newsgroups(subset='test',categories=categories)
x_train=train.data
x_test=test.data
y_train=train.target
y_test=test.target
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
tfidf.fit(x_train)
x_train_=tfidf.transform(x_train)
x_test_=tfidf.transform(x_test)
to_see=pd.DataFrame(x_train_.toarray(),columns=tfidf.get_feature_names_out())
print(to_see.head())
print(to_see.shape)

#建模
from sklearn.naive_bayes import MultinomialNB,ComplementNB,BernoulliNB
from sklearn.metrics import log_loss
name=['multinomial','complement','bernoulli']
models=[MultinomialNB(),ComplementNB(),BernoulliNB()]
for name,clf in zip(name,models):
    clf.fit(x_train_,y_train)
    y_pred=clf.predict(x_test_)
    proba=clf.predict_proba(x_test_)
    score=clf.score(x_test_,y_test)
    print(name)
    log_score=log_loss(y_test,proba)
    print('accuracy:{:.3f}'.format(score))
    print('log loss:{:.3f}'.format(log_score))
    print('\n')
print('*'*60)
#校准来提高模型准确度和降低损失
from sklearn.calibration import CalibratedClassifierCV
name=['multinomial','multinomial+isotonic','multinomial+sigmoid或logistic函数',
      'complement','complement+isotonic','complement+sigmoid或logistic函数',
      'bernoulli','bernoulli+isotonic','bernoulli+sigmoid或logistic函数']
models=[MultinomialNB(),CalibratedClassifierCV(MultinomialNB(),cv=2,method='isotonic'),CalibratedClassifierCV(MultinomialNB(),cv=2,method='sigmoid或logistic函数'),
        ComplementNB(),CalibratedClassifierCV(ComplementNB(),cv=2,method='isotonic'),CalibratedClassifierCV(ComplementNB(),cv=2,method='sigmoid或logistic函数'),
        BernoulliNB(),CalibratedClassifierCV(BernoulliNB(),cv=2,method='isotonic'),CalibratedClassifierCV(BernoulliNB(),cv=2,method='sigmoid或logistic函数')]
for name,clf in zip(name,models):
    clf.fit(x_train_,y_train)
    y_pred=clf.predict(x_test_)
    proba=clf.predict_proba(x_test_)
    score=clf.score(x_test_,y_test)
    print(name)
    log_score=log_loss(y_test,proba)
    print('accuracy:{:.3f}'.format(score))
    print('log loss:{:.3f}'.format(log_score))
    print('\n')
#可以看到，多项式朴素贝叶斯无论怎么调整，算法效果都不如补集朴素贝叶斯来的好，因此我们在分类的时候，应该选择补集朴素贝叶斯