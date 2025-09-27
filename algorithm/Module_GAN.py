import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.util import *



class DiscriminatorNet(torch.nn.Module):

    def __init__(self,
                 chs=[21,9],
                 input_size= 128,
                 dropout=[0.5,0.5],
                 F=32,
                 fc_size = 1024
                 ):
        super().__init__()


        self.sig_conv=nn.Sequential(
            nn.Conv2d(1,F,(2,chs[1]),(2,1)),
            nn.LeakyReLU(0.2),
        )

        with torch.no_grad():
            fake_input = torch.zeros((1, 1, chs[0], input_size))
            fake_output = self.sig_conv(fake_input)
            middle_size = fake_output.shape[2:]

        self.sig_fc=nn.Sequential(
            nn.Linear(int(F * middle_size[0]*middle_size[1]), fc_size),
            nn.LeakyReLU(0.2)
        )

        self.condition_conv=nn.Sequential(
            nn.Conv2d(1,F,(2,chs[1]),(2,1)),
            nn.LeakyReLU(0.2),
        )

        with torch.no_grad():
            fake_input = torch.zeros((1, 1, chs[1], input_size))
            fake_output = self.sig_conv(fake_input)
            middle_size = fake_output.shape[2:]

        self.cond_fc=nn.Sequential(
            nn.Linear(int(F * middle_size[0]*middle_size[1]),fc_size),
            nn.LeakyReLU(0.2)
        )

        self.fully_connected=nn.Linear(2*fc_size,1)



    def forward(self, sig, cond):

        sig=sig.transpose(1,2).reshape(sig.shape[0],1,sig.shape[1],sig.shape[2])
        cond = cond.transpose(1, 2).reshape(cond.shape[0], 1, cond.shape[1], cond.shape[2])

        o1=self.sig_conv(sig)
        o1=torch.reshape(o1,[o1.shape[0],-1])
        o1=self.sig_fc(o1)


        o2=self.condition_conv(cond)
        o2=torch.reshape(o2,[o2.shape[0],-1])
        o2=self.cond_fc(o2)

        o12= torch.cat((o1,o2),dim=1)

        out=self.fully_connected(o12)
        out=torch.sigmoid(out)

        return out


class DiscriminatorNet_Conv(torch.nn.Module):


    def __init__(self,
                 chs=[21,9],
                 Samples= 128,
                 dropout=[0.5,0.5]
                 ):
        super().__init__()

        F1=8
        D=2

        self.sig_module=EEG_Conv(Chans = chs[1],  Samples = Samples,dropoutRate = dropout[0], kernLength = 64, F1 = F1,D = D)
        self.condition_module=EEG_Conv(Chans = chs[0], Samples = Samples,dropoutRate = dropout[1], kernLength = 64, F1 = F1,D = D)

        n_out=39
        self.fully_connected=nn.Linear(int(2*F1*D*n_out),1)



    def forward(self, sig, cond):

        o1=self.sig_module(sig.transpose(1,2))
        o2=self.condition_module(cond.transpose(1,2))


        o12= torch.cat((o1.view(o1.shape[0], -1), o2.view(o2.shape[0],-1)), 1)

        out=self.fully_connected(o12)
        out=torch.sigmoid(out)

        return out
    


class EEG_Conv(nn.Module):


    def __init__(self, Chans = 64,  Samples= 128,
             dropoutRate = 0.5, kernLength = 64, F1 = 8,
             D = 2):
        super(EEG_Conv,self).__init__()

        self.dropout=nn.Dropout(dropoutRate)
        self.MaxValueConstrant=MaxNormConstraint(max_value=1,axis=(1,2,3))


        # conv2d
        # input_size (batch_size,in_channels,eeg_chan,samples)
        # kernel:(in_channels,F1,kernel_size=(1,kernLength)) padding
        # output_size(batch_size,F1,eeg_chan,samples)
        # kernLength = sample_rate / high_pass_frequency
        self.freq_filter_conv=nn.Conv2d(1,F1,(1,kernLength),stride=1,padding=(0, kernLength//2),bias=False)
        self.batch_norm_F1=nn.BatchNorm2d(F1)

        height_out = Chans
        width_out = Samples

        # depthwise_conv
        # input_size (batch_size,F1,eeg_chan,samples)
        # kernel:(F1,D*F1,kernel_size=(eeg_chan,1)) group=D*F1
        # output_size(batch_size,D*F1,1,samples)
        self.depthwise_conv=nn.Conv2d(F1,F1*D,(Chans,1),stride=1,groups=F1,bias=False)
        self.batch_norm_depthwise=nn.BatchNorm2d(F1*D)
        self.pool_depthwise=nn.AvgPool2d(kernel_size=(1,4),stride=(1,4))
        self.depthwise_constran=MaxNormConstraint(max_value=1, axis=(1, 2, 3))
        height_out = compute_conv_outsize(height_out,Chans)
        width_out = compute_conv_outsize(width_out,4,stride=4)

        # seperable_depthwise_conv
        # input_size (batch_size,D*F1,1,samples/4)
        # kernel:(D*F1,F2,kernel_size=(1,kernelLength2)) groups=F2
        # output_size(batch_size,F2,1,width_out)
        # kernelLength2 = time(e.g. 0.5s) * present_rate(e.g. 32Hz)
        #------
        # seperable_pointwise_conv
        # input_size (batch_size,F2,1,width_out)
        # kernel:(F2,F2,kernel_size=(1,1)) groups=F2
        # output_size(batch_size,F2,1,width_out)
        F2=F1*D
        kernelLength2=16
        self.seperable_depthwise_conv=nn.Conv2d(F1*D,F2,(1,kernelLength2),stride=1,
                                                groups=F2,padding=(0, kernelLength2//2),bias=False)
        self.seperable_pointwise_conv=nn.Conv2d(F2,F2,(1,1))
        self.bn_sep_depth=nn.BatchNorm2d(F2)
        self.pool_seperable=nn.AvgPool2d(kernel_size=(1,8),stride=(1,8))

    def forward(self, x):
        # x : batch * 1 * eeg_chan * samples
        x=x.view(-1,1,*x.size()[1:])

        # conv2d
        block0=self.freq_filter_conv(x)
        block0=self.batch_norm_F1(block0)

        #depthwise
        self.depthwise_conv=self.depthwise_constran(self.depthwise_conv)
        block1=self.depthwise_conv(block0)
        block1=self.dropout(block1)
        block1=F.elu(block1)
        block1=self.pool_depthwise(block1)
        block1=self.dropout(block1)


        #separable conv
        #depwise
        block2=self.seperable_depthwise_conv(block1)
        block2=self.seperable_pointwise_conv(block2)
        block2=self.bn_sep_depth(block2)
        block2=F.elu(block2)
        block2=self.pool_seperable(block2)
        out=self.dropout(block2)


        return out

