#Lasso无法解决特征之间精确相关的问题
#当我们使用最小二乘法解决线性回归时，如果线性回归无解或者报除零错误，换Lasso不能解决任何问题，换岭回归处理
#岭回归 vs Lasso
#岭回归可以解决特征间的精确相关关系导致的最小二乘法无法使用的问题，而Lasso不行
#Lasso不是从根本上解决多重共线性的问题，而是限制多重共线性带来的影响（限制w的大小）
#世人其实并不使用Lasso来抑制多重共线性，反而接受了它在其他方面的优势
#Lasso成为了线性模型中特征选择工具的首选
#接下来就看看Lasso如何进行特征选择的