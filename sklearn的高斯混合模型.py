import numpy as np
#生成均值为1.71，标准差为0.056的男生身高数据
np.random.seed(0)
mu_m=1.71 #期望
sigma_m=0.056 #标准差
num_m=10000  #数据个数为10000个
rand_data_m=np.random.normal(mu_m,sigma_m,num_m)  #生成数据
y_m=np.ones(num_m) #生成标签
#生成均值为1.58，标准差为0.051的女生身高数据
mu_w=1.58
sigma_w=0.051
num_w=10000
rand_data_w=np.random.normal(mu_w,sigma_w,num_w)
y_w=np.zeros(num_w)
#把男生和女生数据混合在一起
data=np.append(rand_data_m,rand_data_w)
data=data.reshape(-1,1)
y=np.append(y_m,y_w)

#开始建模
from sklearn.mixture import GaussianMixture
g=GaussianMixture(n_components=2,covariance_type='full',max_iter=1000)#n_components=2是分为多少类,max_iter=1000是迭代多少次
g.fit(data)
print('类别概率',g.weights_) #[0.48215436 0.51784564]
print('均值',g.means_)
print('方差',g.covariances_)
print(g.predict(data))  #分为0或1

#使用高斯混合算法的前提是数据符合高斯分布