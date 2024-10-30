# 1  随机抽取k个样本为最初的质心
# 2  开始循环
# 3  将每个样本点分配到离他们最近的质心，生成k个簇
# 4 对于每个簇，计算所有被分到该簇的样本点的平均值作为新的质心
# 5  当质心位置不再发生变化，迭代停止，聚类完成

#对于一个簇来说，所有样本点到质心的距离越之和小，我们就认为这个簇中的样本越相似
#kmeans追求的是让总体平方和最小的质心
#kmeans比knn慢