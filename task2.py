# -*- coding: UTF-8 -*-
"""
@author:Jaiyaun
@software: PyCharm
@file:task2.py
@time:2021/12/08
"""
# 构建基于 wine quality数据集的回归模型
# （1）根据wine_quality数据集处理的结果,构建线性回归模型。
from task1 import winequality_target_test, winequality_target_train, winequality_trainPca, winequality_testPca
from sklearn.linear_model import LinearRegression

clf = LinearRegression().fit(winequality_trainPca, winequality_target_train)
y_pred = clf.predict(winequality_testPca)
print('线性回归模型预测前10个结果为：', '\n', y_pred[:10])

# (2)根据wine_quality数据集处理的结果,构建梯度提升回归模型。
from sklearn.ensemble import GradientBoostingRegressor

GBR_wine = GradientBoostingRegressor().fit(winequality_trainPca, winequality_target_train)
wine_target_pred = GBR_wine.predict(winequality_testPca)
print('梯度提升回归模型预测前10个结果为：', '\n', wine_target_pred[:10])
print('真实标签前十个预测结果为：', '\n', list(winequality_target_test[:10]))

# (3)结合真实评分和预测评分,计算均方误差、中值绝对误差、可解释方差值。
# (4)根据得分,判定模型的性能优劣
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

print('线性回归模型评价结果：')
print('winequality数据线性回归模型的平均绝对误差为：',
      mean_absolute_error(winequality_target_test, y_pred))
print('winequality数据线性回归模型的均方误差为：',
      mean_squared_error(winequality_target_test, y_pred))
print('winequality数据线性回归模型的中值绝对误差为：',
      median_absolute_error(winequality_target_test, y_pred))
print('winequality数据线性回归模型的可解释方差值为：',
      explained_variance_score(winequality_target_test, y_pred))
print('winequality数据线性回归模型的R方值为：',
      r2_score(winequality_target_test, y_pred))

print('梯度提升回归模型评价结果：')
print('winequality数据梯度提升回归树模型的平均绝对误差为：',
      mean_absolute_error(winequality_target_test, wine_target_pred))
print('winequality数据梯度提升回归树模型的均方误差为：',
      mean_squared_error(winequality_target_test, wine_target_pred))
print('winequality数据梯度提升回归树模型的中值绝对误差为：',
      median_absolute_error(winequality_target_test, wine_target_pred))
print('winequality数据梯度提升回归树模型的可解释方差值为：',
      explained_variance_score(winequality_target_test, wine_target_pred))
print('winequality数据梯度提升回归树模型的R方值为：',
      r2_score(winequality_target_test, wine_target_pred))
