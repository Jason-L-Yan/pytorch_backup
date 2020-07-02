#!/usr/bin/env python
# coding: utf-8
# In[10]:
import torch
from matplotlib import pyplot as plt
from IPython import display
import random


# 图片设置
def use_svg_display():
    """用矢量图显示svg，高清图显示retina"""
    display.set_matplotlib_formats('retina')


# 图片设置
def set_figsize(figsize=(15, 10)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def linreg(x, w, b):
    return torch.mm(x, w) + b


def data_iter(batch_size, features, labels):  # 随机从数据中取出 batch_size的数据，并对所有数据进行遍历
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本读取数据随机
    for i in range(0, num_examples, batch_size):  # 1000个数据遍历一遍
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch_size
        yield features.index_select(0, j), labels.index_select(0, j)  # 0，是按行索引，j是索引的行数。


# 二次代价函数
def square_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size()))**2 / 2


# 梯度更新
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size