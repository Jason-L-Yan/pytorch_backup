from torch.utils.data import TensorDataset
import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.tensor([44, 55, 66])
train_ids = TensorDataset(a, b) 
for x_train, y_label in train_ids:
    print(x_train, y_label)