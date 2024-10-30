#模型评估指标
# 1   普通互信息分
#metrics.adjusted_mutual_info_score(y_pred,y_true)
#调整的互信息分
#metrics.mutual_info_score(y_pred,y_true)           互信息分取值在0到1之间，越接近1，聚类效果越好，在随即均匀聚类下产生0分
#标准化互信息分
#metrics.normalized_mutual_info_score(y_pred,y_true)

# 2  V_measure
#同质性  是否两个簇仅包含单个类的样本
#metrics.homogeneity_score(y_true,y_pred)
#完整性  是否给定类的所有样本都被分配到同一个簇中
#metrics.completeness_score(y_true,y_pred)
#同质性和完整性的调和平均  V_measure                  取值在0到1之间，越接近1，效果越好，对样本分布没有假设，在任何分布都可以有着不错的效果，在随即均匀聚类不会产生0分
#metrics.v_measure_score(y_true,y_pred)
#三者可以被一次性计算出来
#metrics.homogeneity_completeness_v_measure(y_true,y_pred)

#  3 调整兰的系数
#metrics.adjusted_rand_score(y_true,y_pred)  取值在-1和1之间，负值象征着簇内的点差异巨大，甚至相互独立
#正类的兰德系数越接近1越好  对样本分布没有假设，在任何分布上都可以有着不错的表现，尤其是折叠形状德数据  在随即均匀聚类下产生0分


#当真实标签未知的时候，我们用轮廓系数
#它能够同时衡量
#样本与其自身所在的簇中的其他样本相似度a 等于样本与同一簇中所有其他点之间的平均距离
#样本与其他簇中的样本相似度b，等于样本与下一个最近的簇中所有点的平均距离
#根据簇内差异小，簇外差异大 我们希望b永远大于a，并且大的越多越好

#如果一个簇中的大多数样本具有比较高的轮廓系数，则簇会有较高的总轮廓系数，则整个数据集的平均轮廓系数越高，则聚类是合适的
#如果许多样本点具有较低的轮廓系数甚至是负数，则聚类是不合适的，聚类设置的超参数k可能太大或者太小
#传入x y_pred
#metrics.silhouette_score

#当真是标签未知的情况下，除了轮廓系数，还有其他评估指标
#  1  卡林斯基-哈拉巴斯指数
#sklearn.metrics.calinski_harabaz_score(x,y_pred)
#  2  戴维斯-布尔丁指数
#sklearn.metrics.davies_bouldin_score(x,y_pred)
#  3  权变矩阵
#sklearn.metrics.cluster.contingency_matrix(x,y_pred)
#卡林斯基-哈拉巴斯指数  数据之间的离散程度越高，协方差的迹就越大，离散程度越低，协方差的迹就越小，所以我们希望卡林斯基-哈拉巴斯指数越大越好
#卡林斯基-哈拉巴斯指数比起轮廓系数非常快