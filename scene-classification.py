import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

from libs import libsvmDataPreprocessing as DataPreprocessing

if __name__ == '__main__':
    # 获取测试数据集
    df_test = pd.read_csv('dataset/scene_test', delim_whitespace=True, header=None)
    # 获取训练数据集
    df_train = pd.read_csv('dataset/scene_train', delim_whitespace=True, header=None)

    # 获取训练集的标签
    train_target = df_train.iloc[:, 0].values
    train_data = df_train.iloc[:, 1:].values
    # 获取测试集的标签
    test_target = df_test.iloc[:, 0].values
    test_data = df_test.iloc[:, 1:].values

    data_scaler = StandardScaler()

    # 对数据进行预处理
    train_target = DataPreprocessing.preprocessing_target(train_target)
    train_data = DataPreprocessing.preprocessing_data(train_data)
    # 提取训练数据中的均值和方差数据
    train_data_mean = train_data[:, 0:train_data.shape[1]:2]
    train_data_variance = train_data[:, 1:train_data.shape[1]:2]

    test_target = DataPreprocessing.preprocessing_target(test_target)
    test_data = DataPreprocessing.preprocessing_data(test_data)
    # 提取测试数据中的均值和方差数据
    test_data_mean = test_data[:, 0:test_data.shape[1]:2]
    test_data_variance = test_data[:, 1:test_data.shape[1]:2]

    # 对数据进行归一化处理
    train_data_mean_std = data_scaler.fit_transform(train_data_mean)
    train_data_variance_std = data_scaler.fit_transform(train_data_variance)
    test_data_mean_std = data_scaler.fit_transform(test_data_mean)
    test_data_variance_std = data_scaler.fit_transform(test_data_variance)

    # 训练集统计
    train_target_cnt = DataPreprocessing.target_count(train_target)
    # 测试集统计
    test_target_cnt = DataPreprocessing.target_count(test_target)

    # 判断模型文件夹是否为空，如果为空就训练，否则就读取模型
    if not os.listdir('model'):
        # 生成10^-4~10^4次方这个范围内的50个数作为C参数的可能取值
        c_range = np.linspace(1, 2.0, num=50)
        # 生成2^-9~2^3次方这个范围内的13个数作为gamma参数的可能取值
        gamma_range = np.linspace(0, 0.2, num=50)
        # 创建网格搜索的参数
        param_grid = [{
            'C': c_range,
            'gamma': gamma_range
        }]

        # 训练6个分类的SVM模型
        for i in range(len(train_target_cnt)):
            # 循环获取目标统计
            target_cnt = train_target_cnt[i]
            # 采用交叉验证和ova方式，获取匹配和不匹配的数据索引
            clf_index, not_clf_index = DataPreprocessing.get_index_by_clf(target_cnt[0], train_target)

            # 创建一个副本
            train_target_arr = copy.deepcopy(train_target)
            # 修改训练数据集的目标标签
            train_target_arr[clf_index] = target_cnt[0]
            train_target_arr[not_clf_index] = 6
            # 数据类型的转换
            train_target_arr = train_target_arr.astype(np.int)

            # 创建分类器
            scene_svc = SVC(kernel='rbf', class_weight='balanced', probability=True)

            # 创建网格搜索对象，损失函数为accuracy，n_jobs为使用所有cpu
            grid_search = GridSearchCV(scene_svc, param_grid, scoring='accuracy', n_jobs=-1)

            # Step5：训练模型
            grid_result = grid_search.fit(train_data_mean_std, train_target_arr)

            print("%d Target Best: %f using %s" % (i, grid_result.best_score_, grid_search.best_params_))

            # 保存的模型文件
            joblib.dump(grid_search.best_estimator_, 'model/{0}_svc.pkl'.format(target_cnt[0]))

    # 模型列表
    clf_arr = []
    # 获取模型数量
    model_number = len(os.listdir('model'))

    # 载入模型
    for i in range(model_number):
        clf_arr.append(joblib.load('model/{0}_svc.pkl'.format(i)))

    # 保存测试结果
    test_classification = []
    test_probability = []
    test_score = []
    test_distance = []
    test_auc = []
    # 对测试集进行测试
    for i in range(len(test_target_cnt)):
        # AUC参数
        auc_param = {}
        # 循环获取目标统计
        test_cnt = test_target_cnt[i]
        # 采用交叉验证和ova方式，获取匹配和不匹配的数据索引
        clf_index, not_clf_index = DataPreprocessing.get_index_by_clf(test_cnt[0], test_target)

        # 创建一个副本
        test_target_arr = copy.deepcopy(test_target)
        # 修改训练数据集的目标标签
        test_target_arr[clf_index] = test_cnt[0]
        test_target_arr[not_clf_index] = 6
        # 数据类型的转换
        test_target_arr = test_target_arr.astype(np.int)

        # 计算实例到超平面距离
        distance = clf_arr[i].decision_function(test_data_mean_std)
        # 计算测试集精度
        score = clf_arr[i].score(test_data_mean_std, test_target_arr)
        # 计算分类
        classification = clf_arr[i].predict(test_data_mean_std)
        # 计算概率
        probability = np.delete(clf_arr[i].predict_proba(test_data_mean_std), 1, axis=1).flatten()

        test_target_arr_1 = test_target_arr.copy()
        test_target_arr_1[test_target_arr == test_cnt[0]] = 1
        test_target_arr_1[test_target_arr != test_cnt[0]] = 0
        classification_1 = classification.copy()
        classification_1[classification == test_cnt[0]] = 1
        classification_1[classification == 6] = 0
        false_positive_rate, recall, thresholds = roc_curve(test_target_arr_1, classification_1)
        roc_auc = auc(false_positive_rate, recall)
        auc_param['false_positive_rate'] = false_positive_rate
        auc_param['recall'] = recall
        auc_param['thresholds'] = thresholds
        auc_param['roc_auc'] = roc_auc
        test_auc.append(auc_param)

        # 保存2分类的概率，并删除其他概率
        test_probability.append(probability)
        # 保存结果
        test_classification.append(classification)
        test_score.append(score)
        test_distance.append(distance)

    # 获取每个测试实例在每个分类上的概率
    test_probability = np.array(test_probability).T
    # 打印结果
    print(test_classification)
    print(test_score)
    # print(pd.DataFrame(test_probability).head(25))

    for i in range(len(test_auc)):
        plt.subplot(2, 3, i + 1)

        plt.title('%s Class %0.2f' % (i, test_score[i]))
        plt.plot(test_auc[i]['false_positive_rate'], test_auc[i]['recall'], 'b',
                 label='AUC = %0.2f' % test_auc[i]['roc_auc'])
        plt.legend()
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out')

    plt.show()

