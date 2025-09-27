# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/3/20 13:41
import torch
from torch import nn
from algorithm import Constraint
class LSTM(nn.Module):
    '''
        Employ the Bi-LSTM to learn the reliable dependency between spatio-temporal features
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, num_layers=1)

    def forward(self, x):
        b, c, T = x.size()
        x = x.view(x.size(-1), -1, c)  # (b, c, T) -> (T, b, c)
        r_out, _ = self.rnn(x)  # r_out shape [time_step * 2, batch_size, output_size]
        out = r_out.view(b, 2 * T * c, -1)
        return out


class ESNet(nn.Module):
    def calculateOutSize(self, model, nChan, nTime):
        '''
            Calculate the output based on input size
            model is from nn.Module and inputSize is a array
        '''
        data = torch.randn(1, 1, nChan, nTime)
        out = model(data).shape
        return out[1:]

    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block,assign different weight to different channels and fuse them
        '''
        block = []
        block.append(Constraint.Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
                                                     max_norm=1.0))
        block.append(nn.BatchNorm2d(num_features=nChan * 2))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Enhanced structure block,build a CNN block to absorb data and output its stable feature
        '''
        block = []
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride)))
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def __init__(self, num_channels, T, num_classes):
        super(ESNet, self).__init__()
        self.dropout_level = 0.5
        self.F = [num_channels * 2] + [num_channels * 4]
        self.K = 10
        self.S = 2

        net = []
        net.append(self.spatial_block(num_channels, self.dropout_level))
        net.append(self.enhanced_block(self.F[0], self.F[1], self.dropout_level,
                                           self.K, self.S))

        self.conv_layers = nn.Sequential(*net)

        self.fcSize = self.calculateOutSize(self.conv_layers, num_channels, T)
        self.fcUnit = self.fcSize[0] * self.fcSize[1] * self.fcSize[2] * 2
        self.D1 = self.fcUnit // 10
        self.D2 = self.D1 // 5

        self.rnn = LSTM(input_size=self.F[1], hidden_size=self.F[1])

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fcUnit, self.D1),
            nn.PReLU(),
            nn.Linear(self.D1, self.D2),
            nn.PReLU(),
            nn.Dropout(self.dropout_level),
            nn.Linear(self.D2, num_classes))

    def forward(self, x):
        x=x.reshape(x.shape[0],1,*x.shape[1:])
        out = self.conv_layers(x)
        out = out.squeeze(2)
        r_out = self.rnn(out)
        out = self.dense_layers(r_out)
        return out


import numpy as np
import torch.nn.functional as F

class CELoss_Marginal_Smooth(nn.Module):

    def __init__(self, class_num, alpha=0.6, stimulus_type='12'):
        super(CELoss_Marginal_Smooth, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.stimulus_matrix_4 = [[0, 1],
                                  [2, 3]]

        self.stimulus_matrix_12 = [[0, 1, 2, 3],
                                   [4, 5, 6, 7],
                                   [8, 9, 10, 11]]

        if stimulus_type == '4':
            self.stimulus_matrix = self.stimulus_matrix_4

        elif stimulus_type == '12':
            self.stimulus_matrix = self.stimulus_matrix_12


        self.rows = len(self.stimulus_matrix[:])
        self.cols = len(self.stimulus_matrix[0])


        self.attention_lst = [[1.0 / (int(0 <= (i // self.cols - 1) <= self.rows - 1) +
                                      int(0 <= (i // self.cols + 1) <= self.rows - 1) +
                                      int(0 <= (i % self.cols - 1) <= self.cols - 1) +
                                      int(0 <= (i % self.cols + 1) <= self.cols - 1) +
                          int(0 <= (i // self.cols - 1) <= self.rows - 1 and 0 <= i % self.cols - 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols - 1) <= self.rows - 1 and 0 <= i % self.cols + 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols + 1) <= self.rows - 1 and 0 <= i % self.cols - 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols + 1) <= self.rows - 1 and 0 <= i % self.cols + 1 <= self.cols - 1))
                               for j in range(class_num)] for i in range(class_num)]

        self.attention_lst = np.asarray(self.attention_lst)


    def forward(self, outputs, targets):
        '''
        :param outputs: predictive results, shape: (batch_size, class_num)
        :param targets: ground truth, shape: (batch_size,)
        :return:
        '''
        batch_size, class_num = outputs.shape

        # Obtain target labels and create smooth labels
        targets_data = targets.cpu().data
        smoothed_labels = torch.zeros(size=(batch_size, class_num), device=outputs.device)

        # Fill in the remaining parts of the attention matrix
        for i in range(smoothed_labels.shape[0]):
            label = targets_data[i]
            smoothed_labels[i] = torch.from_numpy(self.attention_lst[label])

        # Fill the smooth label with a position assignment of 1.0
        smoothed_labels[torch.arange(batch_size), targets_data] = 1.0

        # Calculate loss
        log_prob = F.log_softmax(outputs, dim=1)
        att_loss = -torch.sum(log_prob * smoothed_labels) / batch_size
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        loss_add = self.alpha * ce_loss + (1 - self.alpha) * att_loss

        return loss_add