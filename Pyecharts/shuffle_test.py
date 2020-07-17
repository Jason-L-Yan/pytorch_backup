import torch
import random
import numpy as np


def data_iter(batch_size, features, labels):  # 随机从数据中取出 batch_size的数据，并对所有数据进行遍历
    num_examples = len(features)
    indices = list(range(num_examples))
    # print("变化前：", indices)
    # random.shuffle(indices)  # 样本读取数据随机
    # print("变化后：", indices)

    for i in range(0, num_examples, batch_size):  # 1000个数据遍历一遍
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch_size
        yield features.index_select(0, j), labels.index_select(0, j)  # 0，是按行索引，j是索引的行数。


num_input = 2  # 即x1与x2两个参数
num_example = 1000  # 数据集样本数，利用np.random.normal生成。
true_w = [2, -3.4]  # 真实值，最后预测值应该无线靠近它
true_b = 4.2  # 真实值，最后预测值应该无线靠近它
# 生成1000行两列的特征数据，这两列，第一列当作x1,第二列当作x2
features = torch.from_numpy(np.random.normal(0, 1, (num_example, num_input)))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 得到1000个数的tensor
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()))  # 在标签中加入噪声，均值为0，方差为0.01。把它当作标准答案，预测值与它相减求平方求二次代价函数

for x, y in data_iter(10, features, labels):
    print("取出的数值：", x)
    print("取出的标签：", y)