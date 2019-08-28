import collections as c
import numpy as np



def knn2(k, predict_point, color_num, feature, label):
    # 根据欧式距离进行计算
    distance = list(map(lambda item: ((item[0] - predict_point) ** 2 + (item[1] - color_num) ** 2) ** 0.5, feature))
    sort_index = np.argsort(distance)
    sorted_label = label[sort_index]
    neighbor = c.Counter(sorted_label[0:k]).most_common(1)[0][0]
    return neighbor


if __name__ == '__main__':
    train_data = np.loadtxt('../data/my_data2_train.csv', delimiter=',')
    feature = train_data[:, 0:2]
    label = train_data[:, -1]
    test_data = np.loadtxt('../data/my_data2_test.csv', delimiter=',')
    for k in range(1, 50):
        count = 0
        for item in test_data:
            predict = knn2(k, item[0], item[1], feature, label)
            if item[-1] == predict:
                count += 1
        chance = (count * 100.0) / len(test_data)
        print('当K为{}，时准确率为{}%'.format(k, chance))
