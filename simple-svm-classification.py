import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.svm import SVC

'''
1、将原始数据转化为SVM算法软件或包所能识别的数据格式；
2、将数据标准化；(防止样本中不同特征数值大小相差较大影响分类器性能)
3、不知使用什么核函数，考虑使用RBF；
4、利用交叉验证网格搜索寻找最优参数(C, gamma)；（交叉验证防止过拟合，网格搜索在指定范围内寻找最优参数）
5、使用最优参数来训练模型；
6、测试。
'''

if __name__ == '__main__':
    # 载入鸢尾花
    iris_dataset = datasets.load_iris()

    feature_names = iris_dataset.feature_names
    target_names = iris_dataset.target_names

    # Step1：获取能够识别的数据格式
    # 获取鸢尾花的特征数据[萼片长度, 萼片宽度, 花瓣长度, 花瓣宽度]
    feature_data = iris_dataset.get('data')
    # 获取鸢尾花的分类结果[刚毛鸢尾, 七彩鸢尾, 维吉尼亚鸢尾]
    target_data = iris_dataset.get('target')

    # Step2：数据标准化
    # 数据标准化，并把计算过程进行保存
    data_scaler = StandardScaler()
    # 得到标准化的输入数据
    x_std = data_scaler.fit_transform(feature_data)

    # 把数据集分为训练集和测试，训练集0.7，测试集0.3
    x_train, x_test, y_train, y_test = train_test_split(x_std, target_data, test_size=0.3)

    # Step3：创建SVM分类器，核函数采用RBF，设置类别权重为n_samples / (n_classes * np.bincount(y))
    iris_svc = SVC(kernel='rbf', class_weight='balanced')

    # Step4：利用交叉验证网格搜索寻找最优参数
    # 生成2^-5~2^15次方这个范围内的11个数作为C参数的可能取值
    c_range = np.logspace(-5, 15, 11, base=2)
    # 生成2^-9~2^3次方这个范围内的13个数作为gamma参数的可能取值
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 创建网格搜索的参数
    param_grid = [{
        'kernel': ['rbf'],
        'C': c_range,
        'gamma': gamma_range
    }]
    # 创建网格搜索对象，其中K-Fold为3，损失函数为accuracy，n_jobs为使用所有cpu
    grid = GridSearchCV(iris_svc, param_grid, scoring='accuracy', cv=3, n_jobs=-1)

    # Step5：训练模型
    grid_result = grid.fit(x_train, y_train)

    # Step6：计算测试集精度
    score = grid.score(x_test, y_test)

    print('精度=%s' % score)



