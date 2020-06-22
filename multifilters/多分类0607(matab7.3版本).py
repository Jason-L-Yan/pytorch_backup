#!/usr/bin/env python
# coding: utf-8

# In[45]:


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import os


# ## 1.训练集数据预处理

# ### 1.1 训练集输入数据

# In[46]:


def data_train(filepath1):  # 训练集数据预处理
    new_datas = list()
    datax = scio.loadmat(filepath1)
    file_x =datax['input_train']
    
    for i in range(file_x.shape[0]):
        for j in range(file_x.shape[1]):
            new_datas.append(complex(file_x[i][j]))
            
    new_datas = np.array(new_datas).reshape(file_x.shape[0],file_x.shape[1])
    new_data_input_r = np.real(new_datas)
    new_data_input_i = np.imag(new_datas)
    datas_all_input = np.column_stack((new_data_input_r, new_data_input_i))
    datas_all_input = torch.tensor(datas_all_input, dtype= torch.float32)
    
    return datas_all_input 


# ### 1.2 训练集标签数据

# In[47]:


def data_train_label(filepath2):  # 训练集数据预处理
    datax = scio.loadmat(filepath2)
    print(datax.keys())
    file_x =datax['chaibiaoqiao_train']
    new_datas = np.array(file_x)
    datas_all_input = torch.tensor(new_datas,dtype= torch.float32)
    
    return datas_all_input 


# ### 1.3 训练集数据参数获取

# In[48]:


class DiabetesDataset(Dataset):
    # __init__中有两种选择
    # 1.把所有数据data，都从init中加载出来，通过__getitem__把数据集中的第i个样本传递过去。
    # 2.数据集比较大时，放置文件名的列表，什么时候用什么时候读取，这样可以保证内存的高效运行
    def __init__(self,filepath1,filepath2):  
        # 因为数据比较小，采用第一种选择，所以将所有数据都加载进了内存中
        #服务器读取数据
        self.x_data = data_train(filepath1)
        self.y_data = data_train_label(filepath2)
        self.len    = (self.y_data).shape[0]
        
        
    def __getitem__(self, index):  # 魔法方法，用于索引数据。
        
        return self.x_data[index], self.y_data[index]  # python中这样写，返回的是一个元组（x_data，y_data）
    
    
    def __len__(self):  # 数据集长度
        
        return self.len


# ## 2. 测试集数据预处理

# ### 2.1 测试集输入数据

# In[49]:


def data_test(filepath3):
    new_datas = list()
    datax = scio.loadmat(filepath3)

    file_x =datax['input_train']
    
    for i in range(file_x.shape[0]):
        for j in range(file_x.shape[1]):
            new_datas.append(complex(file_x[i][j]))
            
    new_datas = np.array(new_datas).reshape(file_x.shape[0],file_x.shape[1])
    new_data_input_r = np.real(new_datas)
    new_data_input_i = np.imag(new_datas)
    datas_all_input = np.column_stack((new_data_input_r, new_data_input_i))
    datas_all_input = torch.tensor(datas_all_input, dtype= torch.float32)
    
    return datas_all_input 


# ### 2.2 测试集标签数据

# In[50]:


def data_test_label(filepath4):  # 训练集数据预处理
    datax = scio.loadmat(filepath4)
    # print(datax.keys())
    file_x =datax['chaibiaoqiao_train']
    new_datas = np.array(file_x)
    datas_all_input = torch.tensor(new_datas,dtype= torch.float32)
    
    return datas_all_input 


# ### 2.3 测试集数据参数获取

# In[51]:


class DiabetesDataset2(Dataset):
    # __init__中有两种选择
    # 1.把所有数据data，都从init中加载出来，通过__getitem__把数据集中的第i个样本传递过去。
    # 2.数据集比较大时，放置文件名的列表，什么时候用什么时候读取，这样可以保证内存的高效运行
    def __init__(self,filepath3,filepath4):  
        # 因为数据比较小，采用第一种选择，所以将所有数据都加载进了内存中
        #服务器读取数据
        self.x_data = data_test(filepath3)
        self.y_data = data_test_label(filepath4)
        self.len    = (self.y_data).shape[0]
        
        
    def __getitem__(self, index):  # 魔法方法，用于索引数据。
        
        return self.x_data[index], self.y_data[index]  # python中这样写，返回的是一个元组（x_data，y_data）
    
    
    def __len__(self):  # 数据集长度
        
        return self.len


# ## 3. 数据调用

# In[52]:
# E:/comic/

dataset  = DiabetesDataset("/home/data_ssd/我的/车载训练数据（多信噪比）/48w组/训练/shuruhe.mat", "/home/data_ssd/我的/车载训练数据（多信噪比）/48w组/训练/chaibiaoqian.mat")
test_dataset = DiabetesDataset2("/home/data_ssd/我的/车载训练数据（多信噪比）/48w组/训练时的测试集/shuruhe.mat", "/home/data_ssd/我的/车载训练数据（多信噪比）/48w组/训练时的测试集/chaibiaoqian.mat")

train_loader = DataLoader(dataset=dataset,
                          batch_size= 500,
                         shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size= 500,
                         shuffle=False)


# ## 4. 模型搭建

# In[53]:



class Module(torch.nn.Module):
    def __init__(self,num_input, num_hiddles1,num_hiddles2,num_hiddles3,num_output):  # define the model architecture
        super(Module, self).__init__()
        self.linear1 = torch.nn.Linear(num_input, num_hiddles1)  # 在句尾多家一个逗号，会报错
        self.linear2 = torch.nn.Linear(num_hiddles1, num_hiddles2)
        self.linear3 = torch.nn.Linear(num_hiddles2, num_hiddles3)
        self.linear4 = torch.nn.Linear(num_hiddles3, num_output)
        # self.sigmoid = torch.nn.Sigmoid()
        
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        return x
    
    
num_input, num_hiddles1,num_hiddles2,num_hiddles3,num_output = 512, 2048,1024,512, 128
model = Module(num_input, num_hiddles1,num_hiddles2,num_hiddles3,num_output)
torch.cuda.empty_cache()
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
loss_fn = torch.nn.BCELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.8)  # 0.5,表示没调用一次 lr 降一半

# ## 5.训练模型

# In[54]:

loss_lists = []
flag = 0
def train(epoch):
    runing_loss = 0.0
    global flag
    global loss_lists
    for i, data in enumerate(train_loader,1) :
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        y_pred = model(inputs)

        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()
        
        runing_loss += loss.item()
        if i % 50  == 0:
            print("[%d, %5d] loss: %.10f" % (epoch+1 , i, runing_loss/50.0))  # 求出每一批次的损失值
            runing_loss = 0.0  # 损失值归零
            # 保存模型

            if len(loss_lists) == 0 or loss< min(loss_lists):
                flag = 0
                torch.save(model.state_dict(), 'class_TwoLayers5.pth')
                # print("best model saved to class_TwoLayers5.pth")
            else:  # 但模型loss没有下降
                # learning rate decay：下降学习率
                # print('min loss',min(loss))
                flag += 1
                if flag == 50:
                    flag = 0
                    scheduler.step()
                    print('@'*80)
                print('%'*80)
        loss_lists.append(loss)
 


# ## 6.测试模型

# In[55]:


def test(epoch):
    correct = 0.0
    total =0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels =data
            images, labels =images.to(device), labels.to(device)
            outputs = model(images)  

            total += labels.cpu().shape[0]  # 每一个batch_size 中labels是一个（N，1）的元组，size(0)=N

            correct +=(torch.round(outputs.cpu()) == torch.round(labels.cpu())).sum().item()  # 对的总个数
    print(total)
#     print(correct)
    if  epoch % 1 ==0:

        print("错误bit数：",total*128.0-correct)
        print("BER: %s " % ((total*128.0-correct)/(total*128.0)))

    if epoch == 999:
        print(outputs[0:5,:])
        print(labels[0:5,:])


# ## 7.启动

# In[56]:


if __name__=="__main__":

    for epoch in range(1000):
        train(epoch)
        # if epoch % 5 ==0:
        test(epoch)
        


# In[ ]:


# print(y_pred[0,0:10])

