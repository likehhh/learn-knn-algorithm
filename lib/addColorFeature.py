import numpy as np


def color_feature(color):
    feature = {"红": 0.50, "黄": 0.51, "蓝": 0.52, "绿": 0.53, "紫": 0.54, "粉": 0.55}
    return feature[color]


if __name__ == "__main__":
    data = np.loadtxt("../data/data1.csv", delimiter=",", encoding="gbk", converters={1: color_feature})
    np.random.shuffle(data)
    np.savetxt("../data/my_data2.csv", data, fmt="%f", delimiter=",")
    np.savetxt("../data/my_data2_train.csv", data[0:100], fmt="%f", delimiter=",")
    np.savetxt("../data/my_data2_test.csv", data[100:-1], fmt="%f", delimiter=",")
