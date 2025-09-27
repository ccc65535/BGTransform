import torch
import torch.nn as nn
import numpy as np
import scipy.signal as signal



class diff_avg_y(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, y):
        # return  torch.mean(torch.pow((out - y), 2))

        batch = y.size(0)
        mean_y = y.mean(1).reshape(batch, 1, -1)
        std_y = y.std(1).reshape(batch, 1, -1)

        loss = torch.mean(torch.abs(out - y) / (mean_y.abs() + 1))

        return loss


class diff_norm_time(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, y):
        # return  torch.mean(torch.pow((out - y), 2))

        batch = y.size(0)
        mean_y = y.mean(1).reshape(batch, 1, -1)
        std_y = y.std(1).reshape(batch, 1, -1)

        y = (y - mean_y) / std_y
        loss_m = torch.mean(torch.abs(out - y))

        return loss_m, mean_y, std_y


class diff_abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, y):
        return torch.mean(torch.abs(out - y))


class diff_pow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, y):
        return torch.mean(torch.pow(out - y, 2))




class diff_stft(nn.Module):
    def __init__(self, srate, win_size):
        super().__init__()
        self.srate = srate
        self.win = win_size

    def forward(self, out, y):
        device = out.device
        out_ = out.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()

        f, t, s1_ = signal.stft(out_, self.srate, nperseg=self.win)
        f, t, s2_ = signal.stft(y_, self.srate, nperseg=self.win)

        s1 = torch.Tensor(s1_.real)
        s2 = torch.Tensor(s2_.real)

        diff1 = torch.mean(torch.pow(out - y, 2))
        diff2 = torch.sum(torch.pow(s1 - s2, 2))
        # print('stft diff:',diff2)

        return diff1 + diff2


class diff_stft2(nn.Module):
    def __init__(self, srate, win_size):
        super().__init__()
        self.srate = srate
        self.win = win_size

    def forward(self, out, y):
        device = out.device
        out_ = out.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()

        f, t, s1_ = signal.stft(out_, self.srate, nperseg=self.win)
        f, t, s2_ = signal.stft(y_, self.srate, nperseg=self.win)

        # s1=torch.Tensor(s1_.real)
        # s2=torch.Tensor(s2_.real)
        s1 = torch.Tensor(s1_.__abs__())
        s2 = torch.Tensor(s2_.__abs__())

        diff1 = torch.mean(torch.pow(out - y, 2))
        diff2 = torch.mean(torch.pow(s1 - s2, 2))
        # print('stft diff:',diff2)

        return diff1 + diff2


class diff_stft3(nn.Module):
    def __init__(self, srate, win_size, a=1, b=10):
        super().__init__()
        self.srate = srate
        self.win = win_size
        self.a = a
        self.b = b

    def forward(self, out, y):
        device = out.device
        ch_num = out.shape[2]

        out_ = out.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()

        stft1 = []
        stft2 = []

        for ch in range(ch_num):
            f, t, s1_ = signal.stft(out_[:, :, ch], self.srate, nperseg=self.win)
            f, t, s2_ = signal.stft(y_[:, :, ch], self.srate, nperseg=self.win)
            stft1.append(s1_.__abs__())
            stft2.append(s2_.__abs__())

        stft1 = torch.Tensor(np.array(stft1))
        stft2 = torch.Tensor(np.array(stft2))

        diff1 = torch.pow(out - y, 2)
        diff2 = torch.pow(stft1 - stft2, 2)

        # diff1 = torch.mean(diff1,axis=(1))
        # diff2 = torch.mean(diff2,axis=(1,2))

        # diff1 = torch.mean(diff1)
        # diff2 = torch.sum(diff2)
        # print(f'diff1:{diff1},diff2:{diff2}')

        diff1 = torch.mean(diff1)
        diff2 = torch.mean(diff2)

        return self.a * diff1 + self.b * diff2


class pearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, y):
        # output=output.detach().cpu()
        # y=y.detach().cpu()
        if len(y.size())==2:
             y=torch.unsqueeze(y,2)
             output=torch.unsqueeze(output,2)
        n_batch, n_sample, n_block = y.size()

        output_mean = torch.mean(output, axis=1)
        y_mean = torch.mean(y, axis=1)
        sumTop = 0.0
        sumBottom = 0.0
        output_pow = 0.0
        y_pow = 0.0
        for i in range(n_sample):
            sumTop += (output[:, i, :] - output_mean) * (y[:, i, :] - y_mean)
        for i in range(n_sample):
            output_pow += torch.pow(output[:, i, :] - output_mean, 2)
        for i in range(n_sample):
            y_pow += torch.pow(y[:, i, :] - y_mean, 2)
        sumBottom = torch.sqrt(output_pow * y_pow)
        loss = (sumTop / sumBottom).abs().mean()
        return loss


### https://github.com/connorlee77/pytorch-mutual-information/blob/master/MutualInformation.py
class MutualInformation(nn.Module):

	def __init__(self, sigma=0.1, num_bins=256, normalize=True):
		super(MutualInformation, self).__init__()

		self.sigma = sigma
		self.num_bins = num_bins
		self.normalize = normalize
		self.epsilon = 1e-10

		self.bins = nn.Parameter(torch.linspace(0, 255, num_bins).float(), requires_grad=False)


	def marginalPdf(self, values):

		residuals = values - self.bins.unsqueeze(0).unsqueeze(0).to(values.device)
		kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
		
		pdf = torch.mean(kernel_values, dim=1)
		normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
		pdf = pdf / normalization
		
		return pdf, kernel_values


	def jointPdf(self, kernel_values1, kernel_values2):

		joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
		normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
		pdf = joint_kernel_values / normalization

		return pdf


	def getMutualInformation(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W

			return: scalar
		'''


		x1 = input1
		x2 = input2
		
		pdf_x1, kernel_values1 = self.marginalPdf(x1)
		pdf_x2, kernel_values2 = self.marginalPdf(x2)
		pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

		H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
		H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
		H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

		mutual_information = H_x1 + H_x2 - H_x1x2
		
		if self.normalize:
			mutual_information = 2*mutual_information/(H_x1+H_x2)

		return mutual_information


	def forward(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W

			return: scalar
		'''
		return self.getMutualInformation(input1, input2)
     


class MutualInformationLoss(nn.Module):
    def __init__(self, num_bins=32):
        """
        初始化互信息损失函数模块。
        :param num_bins: 离散化的直方图的分箱数量。
        """
        super(MutualInformationLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, x, y):
        """
        计算输入 x 和 y 的互信息。
        :param x: 维度为 (batch_size, sample_size) 的输入信号。
        :param y: 维度为 (batch_size, sample_size) 的输入信号。
        :return: 互信息的负值作为损失。
        """
        assert x.shape == y.shape, "x and y must have the same shape"
        

        batch_size = x.size(0)
        mi_loss = 0.0

        for i in range(batch_size):
            x_sample = x[i]
            y_sample = y[i]

            # 计算联合直方图
            x_sample = x[i].detach().cpu().numpy()
            y_sample = y[i].detach().cpu().numpy()

            # 使用 numpy 计算联合直方图
            joint_hist, x_edges, y_edges = np.histogram2d(
                x_sample, y_sample, bins=self.num_bins, range=[[0, 1], [0, 1]]
            )

            # 归一化为联合概率分布
            p_joint = joint_hist / joint_hist.sum()

            # 边缘概率分布
            p_x = p_joint.sum(axis=1, keepdims=True)  # 对 y 轴求和
            p_y = p_joint.sum(axis=0, keepdims=True)  # 对 x 轴求和

            # 避免 log(0) 的数值问题，增加一个小的正数
            eps = 1e-10
            p_joint = p_joint + eps
            p_x = p_x + eps
            p_y = p_y + eps

            # 计算互信息
            mi = np.sum(p_joint * np.log(p_joint / (p_x * p_y)))

            mi_loss -= mi

        return torch.tensor(mi_loss / batch_size, dtype=x.dtype, device=x.device)