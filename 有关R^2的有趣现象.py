import numpy as np
rng=np.random.RandomState(42)
x=rng.randn(100,80)#100行80列
#print(x)
y=rng.randn(100)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
print(cross_val_score(LinearRegression(),x,y,cv=10,scoring='r2'))
#发现R**2会有负数
#因为公式TSS=RSS(解释平方和)+ESS(总离差平方和)不是永远成立的
#R**2衡量的是  1-我们的模型没有捕获到的信息占真实标签中所带的信息量的比例

#当我们的R**2显示为负数的时候，表明我们的模型对我们数据的拟合非常差，模型完全不能使用
#如果出现负数的R**2，先检查建模过程和数据预处理是否正确，也许已经伤害了数据本身
#如果是集成模型的回归，检查弱评估器的数量是否不足，随机森林，梯度提升树这些模型在只有两三颗树的时候很容易出现负数的R**2
#如果检查代码，也确定了数据预处理没有问题，但R**2还是负数，说明线性回归模型不适合数据，试试其他算法