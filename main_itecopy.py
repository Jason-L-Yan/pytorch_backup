import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

from scipy.io import loadmat
# from scipy.linalg import expm, logm
import numpy as np
import math
from one_hot3 import onehot
# import scipy.io as scio
# from scipy.io import loadmat

# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)

CB = loadmat(r"D:\\GitHub\\pytorch_backup\\CB4_6.mat")["CB"]
F2 = loadmat(r"D:\\GitHub\\pytorch_backup\\F4_6.mat")["F"]
num_FN = 4
num_VN = 6
num_M = 4
df = 3
dv = 2
FrameLen = 1024
num_FZ = 5
ITER = 4
Niter_BER = np.zeros((ITER), dtype=int)
# h=np.ones((int(FrameLen/2),num_FN,num_VN))
h = (1 / math.sqrt(2)) * (np.random.randn(int(FrameLen / 2), num_FN, num_VN) + np.random.randn(int(FrameLen / 2), num_FN, num_VN) * 1j)
flag = 0


class MPA(nn.Module):
    def __init__(self):
        super(MPA, self).__init__()
        # self.w0 = nn.Parameter(torch.ones([4, 6]), requires_grad=True)
        self.w0 = nn.Parameter(torch.ones([4, 4, 6]), requires_grad=True)
        # self.w1 = nn.Parameter(torch.ones([4, 6]), requires_grad=True)
        # self.w2 = nn.Parameter(torch.ones([4, 6]), requires_grad=True)
        # self.w3 = nn.Parameter(torch.ones([4, 6]), requires_grad=True)

    def forward(self, num_M, num_FN, num_VN, IVF, VN_index, m, n, FN_index, fa_n):
        global flag
        IFV = np.zeros((num_M, num_FN, num_VN))

        # 更新资源节点
        for VN in range(num_FN):
            for m1 in range(num_M):
                for m2 in range(num_M):
                    for m3 in range(num_M):
                        IFV[m1, VN, int(FN_index[VN, 0])] = \
                            IFV[m1, VN, int(FN_index[VN, 0])] + \
                            IVF[m2, VN, int(FN_index[VN, 1])] * IVF[m3, VN, int(FN_index[VN, 2])] * fa_n[VN, m3, m1, m2]
                        IFV[m1, VN, int(FN_index[VN, 1])] = \
                            IFV[m1, VN, int(FN_index[VN, 1])] + \
                            IVF[m2, VN, int(FN_index[VN, 0])] * IVF[m3, VN, int(FN_index[VN, 2])] * fa_n[VN, m3, m2, m1]
                        IFV[m1, VN, int(FN_index[VN, 2])] = \
                            IFV[m1, VN, int(FN_index[VN, 2])] + \
                            IVF[m2, VN, int(FN_index[VN, 0])] * IVF[m3, VN, int(FN_index[VN, 1])] * fa_n[VN, m1, m2, m3]
        IVF_sum = 0
        IFV = torch.from_numpy(IFV).float()

        if flag == 0:
            flag = 1
            IVF = torch.from_numpy(IVF).float()
        VN_index = torch.from_numpy(VN_index).float()
        if m < n:
            for i in range(num_M):
                IVF_sum = IVF_sum + torch.mul(1 / num_M, IFV[i, :, :])
            for VN in range(num_VN):
                for j in range(num_M):
                    IVF[j, int(VN_index[0, VN]), VN] = torch.mul(1 / num_M, IFV[j, int(VN_index[1, VN]), VN]) / IVF_sum[int(VN_index[1, VN]), VN]
                    IVF[j, int(VN_index[1, VN]), VN] = torch.mul((1 / num_M), IFV[j, int(VN_index[0, VN]), VN]) / IVF_sum[int(VN_index[0, VN]), VN]

            IVF = torch.mul(IVF, self.w0)
            # print(' self.w0' * 10, self.w0)
            # IVF[1] = torch.mul(IVF[1], self.w1)
            # IVF[2] = torch.mul(IVF[2], self.w2)
            # IVF[3] = torch.mul(IVF[3], self.w3)

        return IVF


class MPA2(nn.Module):
    def __init__(self):
        super(MPA2, self).__init__()
        self.wout = nn.Parameter(torch.ones([4, 6]), requires_grad=True)

    def forward(self, num_M, num_VN, IVF, VN_index):
        global flag

        Q = torch.zeros((num_M, num_VN))  # 作为网络隐藏层的最后一层
        for m in range(num_M):  # num_M = 4
            for VN in range(num_VN):  # num_VN = 6
                Q[m, VN] = 1 / num_M * IVF[m, int(VN_index[0, VN]), VN] * IVF[m, int(VN_index[1, VN]), VN]

        # print(num_M, num_VN)
        # Q = torch.sum(1 / num_M * IVF, dim=0) * torch.sum(IVF, dim=1)
        # print(Q.shape)

        # Q = torch.from_numpy(Q).float()
        Q = torch.mul(Q, self.wout)
        # print(' self.wout' * 10, self.wout)
        # print('星期一' * 10)
        flag = 0

        return Q


class MAP_plus(torch.nn.Module):
    def __init__(self):
        super(MAP_plus, self).__init__()
        self.model1 = MPA()
        self.model2 = MPA2()

    def forward(self, num_M, num_FN, num_VN, FN_index, IVF, fa_n, iter, Niter):
        while iter < (Niter + 1):

            # 更新用户节点
            # 应该是需要4个W矩阵；
            # opt.zero_grad()
            IVF = self.model1(num_M, num_FN, num_VN, IVF, VN_index, iter, Niter + 1, FN_index, fa_n)
            iter = iter + 1

        Q = self.model2(num_M, num_VN, IVF, VN_index)
        return Q


adam_lr = 1e-9
network = MAP_plus()
# network2 = MPA2()
model_dict = network.state_dict()

# opt = torch.optim.Adam([{'params': network.parameters(), 'lr': 0.001}, {'params': network2.parameters()}])

opt = optim.Adam(network.parameters(), lr=adam_lr)  # setting for optimizer (Adam)
# opt2 = optim.Adam(network2.parameters(), lr=adam_lr)  # setting for optimizer (Adam)
criterion = nn.MSELoss()
Niter_BER = np.zeros((ITER))
for Niter in range(ITER):

    for FZ in range(num_FZ):

        Array1 = np.zeros((6, FrameLen))
        Array2 = np.zeros((6, FrameLen))
        data_orig = np.random.randint(0, 2, (num_VN, FrameLen))
        origin_matrix = data_orig
        target_matrix = []
        for i in range(len(origin_matrix)):
            target_matrix.append([])
            num = 2
            j = 0
            while j < len(origin_matrix[i]):
                target_matrix[i].append(2 * origin_matrix[i][j] + 1 * origin_matrix[i][j + 1])
                j += num
        target_matrix = np.transpose(target_matrix)
        # print('=' * 100, target_matrix.shape)
        label1 = onehot(target_matrix)
        label1 = label1.reshape(-1, 4)
        # print('=' * 100, label1.shape)
        # ###################SCMA编码#######################
        L_data_orig = int(data_orig.shape[1] / 2)
        data_dec = np.zeros((num_VN, L_data_orig))

        for L in range(L_data_orig):
            data_dec[:, L] = data_orig[:, L * 2] * 2 + data_orig[:, L * 2 + 1]

        # ###################SCMA编码########################
        y = np.zeros((num_FN, L_data_orig), dtype=complex)
        for L in range(L_data_orig):
            for VN in range(num_VN):
                y[:, L] = y[:, L] + CB[:, int(data_dec[VN, L]), VN] * h[L, :, VN]
        data_out = y
        # ##################SCMA编码完毕##################

        EbN0 = 8
        SNR = EbN0 + 10 * math.log10(math.log2(num_M) * num_VN / num_FN)
        N0 = 1 * math.pow(10, -SNR / 10)

        # #################添加高斯白噪声#################
        noise = np.random.randn(2, int(num_FN), int(FrameLen / 2)) * np.tile(N0, (2, int(num_FN), int(FrameLen / 2)))
        data_out_channel = data_out + noise[0, :, :] + noise[1, :, :] * 1j
        # for AWGN in range(data_out.shape[1]):
        #     data_awgn_out = data_out[:, AWGN]
        #
        #     def awgn(x, snr,seed = 7):
        #         snr = 10 ** (snr / 10.0)  # 两个乘号表示乘方
        #         xpower = np.sum(x ** 2) / len(x)
        #         npower = xpower / snr
        #         noise = np.random.randn(len(x)) * np.sqrt(npower)
        #         return x + noise
        #     data_out[:, AWGN] = awgn(data_awgn_out, SNR)

        # #################添加高斯白噪声#################

        # data_out_channel = data_out

        # #########################################################译码译码译码译码译码译码译码译码######################################
        N = data_out_channel.shape[1]
        LLR = np.zeros((num_VN, N * 2))
        for jj in range(N):
            # print("jjjjjjjjjjjj",jj)

            label = label1[6 * jj:6 * (jj + 1), :]
            label = torch.from_numpy(label).float()
            # print("labellabellabellabellabellabel", label)

            FN_index = np.zeros((num_FN, df))
            VN_index = np.zeros((dv, num_VN))
            for VN in range(num_FN):
                ind = np.where(F2[VN, :] == 1)
                FN_index[VN, :] = ind[0]
            for VN in range(num_VN):
                ind = np.where(F2[:, VN] == 1)
                VN_index[:, VN] = ind[0]

            # ############初始化###########
            f = np.zeros((num_FN, num_M, num_M, num_M))
            for FN in range(num_FN):
                for m1 in range(num_M):
                    for m2 in range(num_M):
                        for m3 in range(num_M):
                            f[FN, m3, m1, m2] = -(1 / (2 * N0)) * abs(data_out_channel[FN, jj] - (CB[FN, m1, int(FN_index[FN, 0])] * h[jj, FN, int(FN_index[FN, 0])] + CB[FN, m2, int(FN_index[FN, 1])] * h[jj, FN, int(FN_index[FN, 1])] + CB[FN, m3, int(FN_index[FN, 2])] * h[jj, FN, int(FN_index[FN, 2])])) ** 2

            fa_n = np.exp(f) / math.sqrt(2 * math.pi * N0)
            one = np.ones((num_M, num_FN, num_VN))
            IVF = np.zeros((num_M, num_FN, num_VN))
            for i in range(num_M):
                IVF[i, :, :] = F2 * one[i, :, :]
            IVF = np.dot(1 / num_M, IVF)
            iter = 0
            opt.zero_grad()
            Q = network(num_M, num_FN, num_VN, FN_index, IVF, fa_n, iter, Niter)
            loss = criterion(Q.t(), label)
            loss.backward()
            # for name, parms in network.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
            opt.step()
            # break
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # print('Testing111: ', type(label))
            # label = torch.from_numpy(label)
            # print('3' * 50)
            # label = label.float32()
            # print('$' * 100)
            # ################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # print(Q.t().max(1)[1])
            # print('label: ', label.max(1)[1])

            yuce = Q.t().max(1)[1]
            # print("biaoqianbiaoqianbiaoqian===", biaoqian)

            array1 = np.zeros((6, 2))
            for i in range(len(yuce)):
                a = yuce[i]
                # print("aaa",a)
                b = '{:02b}'.format(a)
                # print("bbb",b[0], b[1])
                array1[i, 0] = b[0]
                array1[i, 1] = b[1]
            # print("aRRAY",array1)
            array2 = np.zeros((6, 2))
            biaoqian = label.max(1)[1]
            for i in range(len(biaoqian)):
                a = biaoqian[i]
                # print("aaa",a)
                b = '{:02b}'.format(a)
                # print("bbb",b[0], b[1])
                array2[i, 0] = b[0]
                array2[i, 1] = b[1]
                # print("aRRAY",array2)
                Array1[:, 2 * jj: 2 * (jj + 1)] = array1
                Array2[:, 2 * jj: 2 * (jj + 1)] = array2
                # print("Array1[:, 2 * jj: 2 * (jj + 1)]", Array1[:, 2 * jj: 2 * (jj + 1)])
                # print("jjjjjj",jj)

            # yuce  = label.max(1)[1]

                # opt.zero_grad()
                # print('你好' * 50)
                # print('$' * 100)
            # label = label.numpy()
            # 输出判决矩阵
            # Q = Q_dec(num_M, num_VN, IVF, VN_index)
        #     # 软判决信息
        #     for VN in range(num_VN):
        #         LLR[VN, 2 * jj] = math.log(Q[0, VN] + Q[1, VN]) - math.log(Q[2, VN] + Q[3, VN])
        #         LLR[VN, 2 * jj + 1] = math.log(Q[0, VN] + Q[2, VN]) - math.log(Q[1, VN] + Q[3, VN])
        # LLR_out = LLR
        # data_out1 = (LLR_out < 0)
        # data_out1 = (data_out1 + 0)
        # Niter_BER[Niter] = Niter_BER[Niter] + len(np.argwhere(data_orig != data_out1))
        # print("FZFZFZFZFZFZ",FZ)

        num_ber = len(np.argwhere(Array1 != Array2))
        # print("Array1Array1Array1Array1", Array1.shape)
        # print("label1label1label1label1", label1.shape)
        # print(num_ber)
        Niter_BER[Niter] = num_ber / (6 * FrameLen * num_FZ) + Niter_BER[Niter]
        print("Niter_BERNiter_BERNiter_BER", Niter_BER)
        # if Niter == 3:
        #     if FZ == 2:
        #         yuce_date = r'C:\Users\ghy15236621158\Desktop\Pycharm\date\yuce_date.mat'
        #         scio.savemat(yuce_date, {'yuce_date': Array1})
        #         biaoqian_date = r'C:\Users\ghy15236621158\Desktop\Pycharm\date\biaoqian_date.mat'
        #         scio.savemat(biaoqian_date, {'biaoqian_date': Array2})
    print("Niter", Niter)