#criterion
#输入mse使用均方误差
#输入friedman_mse使用费尔德曼均方误差
#输入mae使用绝对平均误差

#在回归树中score返回的是R平方，不是mse
#在score中输入scoring='neg_mean_squared_error'返回的是负均方误差
#R平方的取值范围是负无穷到1