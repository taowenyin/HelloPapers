import numpy as np
import pandas as pd
import time
import sklearn.tree as tree
import graphviz
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from libs import libsvmDataPreprocessing as DataPreprocessing

if __name__ == '__main__':
    # 计算开始时间
    star = time.time()

    # 创建数据集的特征名
    feature_names = np.arange(1, 104)
    # 创建数据集的目标名
    target_names = np.arange(0, 14)

    # 获取测试数据集
    df_test = pd.read_csv('dataset/yeast_test.svm', delim_whitespace=True, header=None)
    # 获取训练数据集
    df_train = pd.read_csv('dataset/yeast_train.svm', delim_whitespace=True, header=None)

    # 获取训练集的标签
    train_target = df_train.iloc[:, 0].values
    train_data = df_train.iloc[:, 1:].values
    # 获取测试集的标签
    test_target = df_test.iloc[:, 0].values
    test_data = df_test.iloc[:, 1:].values

    # 创建数据标准化对象
    data_scaler = StandardScaler()
    # 创建多标签二值化对象
    multiLabelBinarizer = MultiLabelBinarizer()
    # 对训练数据进行主成分分析
    # 设置主成分属性n_components为mle，即自动选取特征个数，使得满足所要求的方差百分比
    # 设置copy为True，将原始训练数据复制
    # 设置whiten为False，不需要进行特征具有相同的方差
    # pca = PCA(n_components='mle', copy=True, whiten=False)
    pca = PCA(n_components=2, copy=True, whiten=False)

    # 对数据进行预处理
    train_target = DataPreprocessing.preprocessing_target(train_target)
    # 把标签进行多标签二值化
    train_target = multiLabelBinarizer.fit_transform(train_target)
    train_data = DataPreprocessing.preprocessing_data(train_data)
    # # 把数据进行归一化
    # train_data = data_scaler.fit_transform(train_data)
    # # 通过主成分分析对数据进行降维
    # train_data = pca.fit_transform(train_data)

    test_target = DataPreprocessing.preprocessing_target(test_target)
    test_target = multiLabelBinarizer.transform(test_target)
    test_data = DataPreprocessing.preprocessing_data(test_data)
    # test_data = data_scaler.fit_transform(test_data)
    # test_data = pca.fit_transform(test_data)

    # 创建决策树分类器
    # 设置随机种子random_state为0，保证每次分割训练集和测试集的方式都相同
    # 设置分割点splitter为best，使得每次分割都采用最佳分割
    clf = DecisionTreeClassifier(random_state=0, splitter='best', class_weight='balanced')
    clf.fit(train_data, train_target)

    # # 到处决策树结构的DOT数据
    # dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names,
    #                                 class_names=target_names, filled=True, rounded=True,
    #                                 special_characters=True)
    # # 从DOT创建Graph对象，并把树结构保存为PDF
    # graph = graphviz.Source(dot_data)
    # graph.view(filename='tree-graph')

    # 对策数据进行预测
    predict_target = clf.predict(test_data)

    # 测试集精度s
    accuracy_score = accuracy_score(test_target, predict_target)
    print('测试集精度={:.2f}%'.format((1-accuracy_score) * 100))

    # 获取所有特征的重要程度
    feature_importance = clf.feature_importances_

    # 绘制特征重要程度的柱状突
    plt.title('Feature Importance')
    plt.bar(feature_names, feature_importance)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.show()

    # 计算结束事时间
    end = time.time()
    print('用时：{:.3f}s'.format(end - star))
