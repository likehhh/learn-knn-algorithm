import numpy as np
import collections as c


def knn(k, predict_point, feature, label):
    # 计算每个投掷点距离predictPoint的距离
    distance = list(map(lambda x: abs(predict_point - x), feature))
    # 对distance的集合 元素从小到大排序（返回的是排序的下标位置)
    sort_index = (np.argsort(distance))
    # 用排序的sort_index来操作 label集合
    sorted_label = (label[sort_index])
    return c.Counter(sorted_label[0:k]).most_common(1)[0][0]


if __name__ == '__main__':
    train_data = np.loadtxt("../data/data0-train.csv", delimiter=",")
    feature = (train_data[:, 0])
    label = train_data[:, -1]
    test_data = np.loadtxt('../data/data0-test.csv', delimiter=',')
    # k = 4
    for k in range(1, 100):
        count = 0
        for item in test_data:
            # print("投掷点为：{}时，检测结果为：{}，正确结果为：{}".format(item[0], knn(k, item[0], feature, label), item[1]))
            predict = knn(k, item[0], feature, label)
            if predict == item[1]:
                count += 1
        chance = (count * 100.0) / len(test_data)
        print("k为：{}时，准确率为：{}".format(k, chance))
        # print("准确率为：{}%".format(chance))
