#为了升维找到超平面，即使已知映射函数，计算量可能也会无比的大，要找出超平面的代价也是非常大的
#为了解决这个问题，有一个数学方式叫核技巧（kernel trick）是一种能够使用数据原始空间中的向量计算来表示升维后的空间中的点积结果的数学方式
#而这个原始空间中的点积，就被叫做核函数

#参数kernel  的取值
#输入  linear 线性核 解决线性问题
#输入  poly  多项式核  解决偏线性问题  有参数gamma  degree  coef0
#输入  sigmoid或logistic函数  双曲正切核  解决非线性问题   有参数gamma  coef0
#输入  rbf   高斯径向基  解决偏非线性问题  有参数gamma

#degree  填整数，可不填，默认为3
#多项式核函数的次数，如果核函数中没有选择poly，这个参数会被忽略

#gamma  填浮点数  可不填  默认auto
#输入auto 自动使用1/(n_features)作为gamma的值
#输入scale  则使用1/(n_features*x.std())作为gamma的值

#coef0  浮点数 可不填，默认0.0
#核函数中获得常数项
