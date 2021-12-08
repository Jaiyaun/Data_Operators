# -*- coding: UTF-8 -*-
"""
@author:Jaiyaun
@software: PyCharm
@file:task4.py
@time:2021/12/08
"""
# 构建基于wine数据集的K-Means聚类模型
# 1、根据实训1的wine数据集处理的结果,构建聚类数目为3的 K-Means模型
from task1 import wine_target_train, wine_trainScaler, wine
from sklearn.cluster import KMeans

# 用标准化后的训练集建模
kmeans = KMeans(n_clusters=3, random_state=1).fit(wine_trainScaler)
# 用标准化后PCA降维后的训练集建模(采用降维后的数据聚类效果不好，故此处不采用)
# kmeans = KMeans(n_clusters = 3,random_state=1).fit(wine_trainPca)
print('构建的KMeans模型为：\n', kmeans)

# 2、对比真实标签和聚类标签求取FMI
from sklearn.metrics import fowlkes_mallows_score  # FMI评价法

score = fowlkes_mallows_score(wine_target_train, kmeans.labels_)
print("wine数据集的FMI:%f" % (score))

# 3、在聚类数目为2~10类时,确定最优聚类数目
for i in range(2, 11):
    ##构建并训练模型
    kmeans = KMeans(n_clusters=i, random_state=123).fit(wine_trainScaler)
    score = fowlkes_mallows_score(wine_target_train, kmeans.labels_)
    print('wine数据聚%d类FMI评价分值为：%f' % (i, score))

# 4、求取模型的轮廓系数,绘制轮廓系数折线图,确定最优聚类数目
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouettteScore = []
for i in range(2, 11):
    # 构建并训练模型
    kmeans = KMeans(n_clusters=i, random_state=1).fit(wine)
    score = silhouette_score(wine, kmeans.labels_)
    silhouettteScore.append(score)
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouettteScore, linewidth=1.5, linestyle="-")
plt.show()

# 5、求取 Calinski-Harabasz指数,确定最优聚类数
from sklearn.metrics import calinski_harabasz_score

for i in range(2, 11):
    # 构建并训练模型
    kmeans = KMeans(n_clusters=i, random_state=1).fit(wine_trainScaler)
    score = calinski_harabasz_score(wine_trainScaler, kmeans.labels_)
    print('seeds数据聚%d类calinski_harabaz指数为：%f' % (i, score))
