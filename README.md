# MLADV_finalpj

- 进度
  - PPCA EM(complete 100%)
  - Kernel PCA EM(complete 100% -> negative eigen value when using eig)
  - Bayesian PCA(complete 99% -> speed)
  - Evaluation method(on going 50% -> 1-NN GE? Kmean-like method? 2d visulization? inflection point? M determine? classification? real world data?)
  - Essay framework discussion(tomorrow)
  - Essay(0%)

1. ppca_weng 包含两个版本的EM PPCA，simple_PPCA类和PPCA。EM停止迭代条件严格来说应是判断参数或者Q收敛与否，这里暂时只是简单设置迭代次数，simple PCPA是10000次，PPCA是迭代100次。

  + simple_PPCA: 不考虑sigma，在EM迭代更新中，运算皆能使用矩阵运算进行加速。
  + PPCA: 考虑了sigma的EM PPCA，由于在更新sigma的时候，其中有一项需要根据对应不同点计算迹，且其中包含矩阵乘法无法进行分解提取，因此需要单独计算。
  
2. 使用的数据集是scipy自带的四个toy dataset，分别是iris，digits，wine和breast_cancer。

3. 测试指标使用generalization error，参考于[Dimensionality Reduction: A Comparative Review](https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf)。
另外两个指标trustworthiness, continuity暂时不用（计算稍微复杂）。

4. 原数据维度为D，保留的主成分从1逐渐增加到D，最高为20。

5. 运行方法为 python finalpj.py [id]。 使用simple PPCA，迭代1000次。可将73，74行代码更改为使用general的PPCA。
```
pp = PPCA()
# pp = simple_PPCA()
```

6. 运行示意图如下，使用digit数据集。
![](figure/example_digits.png)

7. 待修改：
  + 加入real world dataset(400x4069): The Olivetti faces dataset
  + 加入Mnist (70000x784)（数据集太大，算GE的时候太慢)
  + Generalization error 的计算需要再斟酌一下，是否需要使用测试集
