#支持向量机的分类方法，是在这组分布中找出一个超平面作为决策边界，是模型在数据分类误差尽量小，尤其是在未知数据集上的分类误差（泛化误差）尽量小

#超平面
#超平面是一个空间的子空间，它是维度比所在空间小一维的空间，如果数据本身的空间是三维的，则其超平面是二维的，如果数据空间本身是二维的，则其超平面是一维的直线
#在二分类问题中，如果一个超平面能够将数据划分为两个集合，则其每个集合中包含单独的一个类别，我们就说这个超平面是数据的决策边界

#我们把决策边界B1向两边平移，知道碰到离这条决策边界最近的数据点为止，形成两个新的超平面，分别是b11  b12，并且我们将原始的决策边界移动到b11 b12的中间
#直到决策边界到b11 b12的距离相等，在b11和b12中间的距离，叫做B1这条决策边界的边际，通常记作d

#边际很小时，是一种在训练集表现很好的情况，在测试集上表现糟糕，会过拟合
#所以我们在寻找决策边界时，希望边际越大越好

#支持向量机，就是通过找出边际最大的决策边界，来对数据进行分类的分类器