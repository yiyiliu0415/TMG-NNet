import torch
import torch.nn as nn
from TMGnet import Small_NN
import time
import scipy.io as scio
import numpy as np
import h5py
from scipy.io import loadmat

learning_rate = 1e-4
epoch = 1000
split_epoch = epoch // 1.5
M = 1600
N = 3600
sample_size = (N, M)
total_size = sample_size[0] * sample_size[1]
gamma = 8
data_num = N * gamma

lamda1 = 15
lamda2 = 1

out_path = r'.\matrix\out_mat_40.mat'
slm_path = r'.\matrix\slm_mat2.mat'

out_mat = loadmat(out_path)
slm_mat = loadmat(slm_path)

Eout = out_mat['out_mat_40'][:data_num]
slm = slm_mat['slm_mat'][:]

print('loaded Eout, shape:', Eout.shape)
print('loaded Ein, shape:', slm.shape)

Eout = torch.tensor(Eout).to('cuda').float()
Eout = Eout / 255.
slm = torch.tensor(slm).to('cuda').float()
slm = torch.exp(1j * 2 * torch.pi * slm)


def cal_var(x):
    mean = torch.mean(x)
    var = torch.mean((x - mean) ** 2)
    return var


class TVLoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super(TVLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, x):
        height, width = x.size()

        tv_h = torch.pow(x[:, 1:] - x[:, :-1], 2).sum()

        tv_w = torch.pow(x[1:, :] - x[:-1, :], 2).sum()

        tv_loss = tv_h + tv_w

        if self.reduction == 'mean':
            tv_loss = tv_loss / (height * width)
        elif self.reduction == 'sum':
            pass

        return self.weight * tv_loss


def guassian_loss(out):
    real = out.real
    imag = out.imag
    var_real = cal_var(real)
    var_imag = cal_var(imag)
    loss = (lamda1 / var_real) * (torch.mean(real ** 2)) + (lamda1 / var_imag) * (torch.mean(imag ** 2))
    return loss


def new_loss(out, true_Eo):
    E_o = torch.matmul(slm, out)
    I_o = torch.abs(E_o)
    I_o = I_o ** 2
    loss = (I_o - true_Eo) ** 2
    loss = torch.mean(loss)
    return loss


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every ? epochs"""
    lr = learning_rate * (0.1 ** (epoch // 250))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model_r = Small_NN(total_size, total_size).to(device)
model_i = Small_NN(total_size, total_size).to(device)
optimizer_i = torch.optim.Adam(model_i.parameters(), lr=learning_rate, weight_decay=4e-5)
optimizer_r = torch.optim.Adam(model_r.parameters(), lr=learning_rate, weight_decay=4e-5)


def train_model():
    model_r.train()
    model_i.train()
    model_r_out = model_r(NN_input)
    model_i_out = model_i(NN_input)

    model_out = torch.complex(model_r_out, model_i_out)
    model_out = torch.reshape(model_out, [sample_size[0], sample_size[1]])

    loss_model = new_loss(model_out, Eout) + lamda1*guassian_loss(model_out) + lamda2*TVLoss()(model_out)

    optimizer_r.zero_grad()
    optimizer_i.zero_grad()
    loss_model.backward()
    optimizer_r.step()
    optimizer_i.step()

    return loss_model


def get_G(NN_input):
    model_r.eval()
    model_i.eval()
    model_r_out = model_r(NN_input)
    model_i_out = model_i(NN_input)

    pred_TM = torch.complex(model_r_out, model_i_out)
    pred_TM = torch.reshape(pred_TM, [sample_size[0], sample_size[1]])
    return pred_TM


if __name__ == '__main__':
    start = time.time()
    init_loss = 0
    loss_list = []
    all_G = []
    try:
        print('Start training.')

        NN_input = torch.rand([1, total_size]).to(device)

        for i in range(epoch + 1):
            start1 = time.time()
            train_loss = train_model()
            adjust_learning_rate(optimizer_r, i)
            end1 = time.time()
            print(f'Training time for epoch {i}: {end1 - start1} seconds.')

        loss_list.append(train_loss.detach().cpu().numpy())

        print(f'Training finished.')

    except KeyboardInterrupt:
        print('Training interrupted.')

    pred_G = get_G(NN_input).detach().cpu().numpy()
    np.save(f'data/TMG.npy', pred_G)
    np.save(f'data/losses.npy', loss_list)
    scio.savemat(r'.\matrix\pred_slm_whole100.mat', {'tmg_tm': pred_G})
    end = time.time()
    print('Time:', end - start)
