# -*- coding: UTF-8 -*-
"""
@author:Jaiyaun
@software: PyCharm
@file:task3.py
@time:2021/12/08
"""
# 构建基于wine数据集的SVM分类模型
# (1)读取wine数据集,区分标签和数据
import pandas as pd

wine = pd.read_csv('data/wine.csv')
wine_data = wine.iloc[:, 1:]
wine_target = wine['Class']

# (2)将wine数据集划分为训练集和测试集
from sklearn.model_selection import train_test_split

wine_data_train, wine_data_test, wine_target_train, wine_target_test = \
    train_test_split(wine_data, wine_target, test_size=0.1, random_state=6)

# (3)使用离差标准化方法标准化wine数据集。
from sklearn.preprocessing import MinMaxScaler  # 标准差标准化

stdScale = MinMaxScaler().fit(wine_data_train)  # 生成规则（建模）
wine_trainScaler = stdScale.transform(wine_data_train)  # 对训练集进行标准化
wine_testScaler = stdScale.transform(wine_data_test)  # 用训练集训练的模型对测试集标准化

# (4)构建SVM模型,并预测测试集结果。
from sklearn.svm import SVC

svm = SVC().fit(wine_trainScaler, wine_target_train)
print('建立的SVM模型为：\n', svm)
wine_target_pred = svm.predict(wine_testScaler)
print('预测前10个结果为：\n', wine_target_pred[:10])

# (5)打印出分类报告,评价分类模型性能
from sklearn.metrics import classification_report

print('使用SVM预测iris数据的分类报告为：', '\n',
      classification_report(wine_target_test,
                            wine_target_pred))
