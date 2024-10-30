#liner_model.LinearRegression

#fit_intercept  填布尔值，可不填，默认为true，是否计算此模型的截距
#normalize  填布尔值，可不填，默认为false
#如果为true。则特征矩阵x在进入回归之前将会被减去均值（中心化）并除以l2范式（缩放），如果希望进行标准化，请使用专门的类preprocessing.StandardScaler
#copy_X  填布尔值  可不填，默认为true  如果为真，将在x.copy()上进行计算，否则的话原本的特征矩阵x可能被线性回归影响并覆盖
#n_jobs  使用计算的cup 默认为1  填-1表示使用所有的cpu进行计算

#线性回归参数太简单，没有调参的余地

#调用coef_  查看系数向量
#调用intercept_  查看截距