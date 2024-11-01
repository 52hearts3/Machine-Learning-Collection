#假设特征之间是有条件独立的，可以解决众多问题，也简化了许多计算过程，这也是朴素贝叶斯被称为朴素的理由
#因此，贝叶斯在特征之间有较多的相关性的数据集上表现不佳
#而现实数据多多少少都会有一些相关性，所以贝叶斯的分类效力在分类算法中不算特别强大
#同时，一些影响特征本身相关性的降维算法，如pca和SVD，和贝叶斯连用效果也会不佳