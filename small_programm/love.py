# from pyecharts.charts import Bar  # 导入bar模块
# attr = ["音响", "电视", "相机", "Pad", "手机", "电脑"]  # 设置x轴数据
# v1 = [5, 20, 36, 10, 75, 90]  # 第一组数据
# v2 = [10, 25, 8, 60, 20, 80]  # 第二组数据
# bar = Bar()  # 实例一个柱状图#
# # bar._use_theme("macarons")  # 指定图表显示的主题风格，后面会讲
# bar.add_xaxis(attr)
# bar.add_yaxis("京东", v1)  # 用add函数往图里添加数据并设置is_stack为堆叠
# bar.add_yaxis("淘宝", v2)  # mark_point标记min,max,average, mark_line标记线
# bar.render("1.1.柱状图数据堆叠示例33.html")  # 保存为html类型


import torch
import numpy as np
from matplotlib import pyplot as plt
import random
import sys
sys.path.append(r'./')  # 我把d21zh_pytorch.py文件定义在本程序所在文件夹的同一级
from d21zh_pytorch import*

num_input = 2  # 即x1与x2两个参数
num_example = 1000  # 数据集样本数，利用np.random.normal生成。
true_w = [2, -3.4]  # 真实值，最后预测值应该无线靠近它
true_b = 4.2  # 真实值，最后预测值应该无线靠近它
# 生成1000行两列的特征数据，这两列，第一列当作x1,第二列当作x2
features = torch.from_numpy(np.random.normal(0, 1, (num_example, num_input)))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 得到1000个数的tensor
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()))  # 在标签中加入噪声，均值为0，方差为0.01。把它当作标准答案，预测值与它相减求平方求二次代价函数
print(features[0], labels[0])
set_figsize()
plt.tick_params(axis='x', colors='red')  # 设置坐标轴刻度颜色
plt.tick_params(axis='y', colors='red')  # 设置坐标轴刻度颜色
plt.xlabel('features[:,1]', color='red', fontdict={'size': 15})  # 设置横坐标标签
plt.ylabel('labels', color='red', fontdict={'size': 15})
# 画出特征1与标签的关系图
plt.scatter(features[:, 1].numpy(), labels.numpy())
# 需要显示图片时，删除注释
# plt.show()  

batch_size = 10
w = torch.tensor(np.random.normal(0, 0.01, (num_input, 1)), dtype=torch.float64, requires_grad=True)  # 对将要进行预测的w值，进行初始化。因为features是每一行两列，根据矩阵计算原则，w是两列一行
b = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # b是一个标量，就初始化成一个数0。

lr = 0.03  # 学习速率
num_epochs = 10  # 迭代次数
net = linreg  # 在上面的函数中被定义，这里调用
loss = square_loss  # 在上面的函数中被定义，这里调用

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        L = loss(net(x, w, b), y).sum()
        L.backward()
        sgd([w, b], lr, batch_size)
        w.grad.data.zero_()  # 梯度归零。每次梯度必须归零。因为在pytorch中梯度是累加的，归零之后，上次的梯度不会影响这一次的
        b.grad.data.zero_()
    train_1 = loss(net(features, w, b), labels)  # 一次迭代完成后得到的代价函数
    print('epoch %d ,loss %f ' % (epoch + 1, train_1.mean().item()))
print(true_w, '\n', w)  # 查看预测值，与实际值是否很相近
print(true_b, '\n', b)         
print("nihao")    


