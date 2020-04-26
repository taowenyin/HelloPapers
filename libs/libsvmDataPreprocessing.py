import numpy as np


# 数据预处理，结果为N维的多数组
def preprocessing_data(data):
    # 保存最终的处理数据
    train_data_arr = []
    for i in range(len(data)):
        # 获取每个数据进行处理
        item = data[i]
        item_arr = []
        for j in range(len(item)):
            sub_item = np.array(item[j].split(':')).astype(np.float)
            item_arr.append(sub_item[1])

        train_data_arr.append(np.array(item_arr))
    # 构建标签集
    return np.array(train_data_arr)


# 标签预处理，结果为N维的多数组
def preprocessing_target(target):
    # 标签集预处理
    train_target_arr = []
    for i in range(len(target)):
        item = target[i]
        if len(item) > 1:
            arr = np.array(item.split(',')).astype(np.int)
        else:
            arr = np.array([int(item)])

        # 标签进行排序
        arr = np.array(sorted(arr))
        train_target_arr.append(arr)
    # 构建标签集
    return np.array(train_target_arr)


# 标签统计
def target_count(target):
    target_cnt = {}
    for i in range(len(target)):
        targets = target[i]
        for j in range(len(targets)):
            # 如果Key值不存在，那么设置数量为1，否则+1
            target_cnt[targets[j]] = target_cnt.get(targets[j], 0) + 1

    # 按照标签key统计重新排序
    target_cnt = sorted(target_cnt.items(), key=lambda x: x[0])

    return target_cnt


# 根据类别获取数据响应的索引
def get_index_by_clf(clf, target_data):
    # 保存匹配的索引
    index_equ_arr = []
    # 保存不匹配的索引
    index_not_equ_arr = []

    for i in range(len(target_data)):
        # 找到匹配的索引
        is_find = False
        target_item = target_data[i]
        for j in range(len(target_item)):
            item = target_item[j]
            # 判断当前数据是否有检索的分类标签
            if clf == item:
                # 假如有分类标签，那么就把该数据的父索引假如数据
                index_equ_arr.append(i)
                is_find = True
                break
        # 保存不符合要求的索引
        if not is_find:
            index_not_equ_arr.append(i)

    return np.array(index_equ_arr), np.array(index_not_equ_arr)
