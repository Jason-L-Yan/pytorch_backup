from torch.utils.data import TensorDataset
import torch
"""TensorDataset 可以用来对 tensor 进行打包，就好像 python 中的 zip 功能。该类通过每一个 tensor 的第一个维度进行索引。因此，该类中的 tensor 第一维度必须相等。
注意：TensorDataset 中的参数必须是 tensor"""
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.tensor([44, 55, 66])
train_ids = TensorDataset(a, b)

print(train_ids[0:2])  # 切片
print('=' * 80)
for x_train, y_label in train_ids:
    print(x_train, y_label)
