import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.manifold import TSNE
from sympy import rotations 

from augmentation import BGTransform as bgt

from lib.LoadSet import *

from sklearn.model_selection import KFold

from augmentation.base import augment
from scipy.fftpack import fft

def FFT(Fs, data):
    """
    对输入信号进行FFT
    :param Fs:  采样频率
    :param data:待FFT的序列
    :return:
    """
    L = len(data)  # 信号长度
    N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂，也即N个点的FFT
    result = np.abs(fft(x=data, n=int(N))) / L * 2  # N点FFT
    axisFreq = np.arange(int(N / 2)) * Fs / N  # 频率坐标
    result = result[range(int(N / 2))]  # 因为图形对称，所以取一半
    return axisFreq, result

srate = 250
root_dir = 'D:\\EEGData\\MetaBCI\\MNE-nakanishi2015-data\\mnakanishi\\12JFPM_SSVEP\\raw\\master\\data\\'
eeg_all=load_Nak(root_dir,srate)
eeg_all


srate = 250

offLength = 0.135 
start =int(srate*offLength)

duration=1
k=0
#1,3,4,5,7,9
sub_id=1
    

stop = start + int(srate*duration)

eeg=eeg_all[:,start:stop,:,:,:]
eeg-=np.mean(eeg,axis=1,keepdims=True)
nchannel,nsample,nclass,ntrial,nsubject=eeg.shape
folds=range(ntrial)
trials=[t for t in range(ntrial)]


n_splits=15


list(set(trials).difference([k]))

sub_data=eeg[:,:,:,:,sub_id]

nch,nsamp,nclass,ntrail=sub_data.shape


sub_trainX,sub_trainY=[],[]
for c in range(nclass):
    for t in range(ntrail):
        sub_trainX.append(sub_data[:,:,c,t])
        sub_trainY.append(c)
sub_trainX=np.array(sub_trainX)
sub_trainY=np.array(sub_trainY)


aug_names=['TimeReverse','SmoothTimeMask','FrequencyShift','FTSurrogate']

colors=[
    'xkcd:cerulean',
    'xkcd:grey',
    'xkcd:steel grey',
    'xkcd:blue'
]

y1=0
ch=1
for type in aug_names:
    ind=np.where(sub_trainY==y1)
    block_X=sub_trainX[ind]
    block_y=sub_trainY[ind]

    # if type=='FTSurrogate':
    #     flag=True
    # else:
    #     flag=False

    k=2
    trials=block_X[:k,:,:]
    
    # trials_t,_=augment(type,trials,np.ones(trials.shape[0]),p=1,srate=srate)

    template=np.mean(trials,axis=0,keepdims=True)
    bg=trials-template.repeat(k,0)

    bg_t,_=augment(type,bg,np.ones(bg.shape[0]),p=1,srate=srate)
    bg_t=bg_t.numpy()
    gen_trials=template+bg_t

    plt.figure(figsize=(8,6))
    # plt.title(type)

    sub_fig=plt.subplot(4,2,1)
    sub_fig.plot(trials[0][ch],c=colors[0])
    sub_fig.set_xlim(0,250)
    sub_fig.set_xticks([0,125,250])
    sub_fig.set_xticklabels([0,0.5,1])
    # sub_fig.set_ylabel('Sample',rotation=0, labelpad=25)
    sub_fig.set_title('Sample',x=-.25,y=.3)
    # sub_fig.set_xlabel('Time(s)')


    sub_fig=plt.subplot(4,2,2)
    freq,spec=FFT(srate,trials[0][ch])
    sub_fig.plot(freq,spec,c=colors[0])
    sub_fig.set_xlim(0,20)
    # sub_fig.set_xlabel('Frequency(Hz)')
    

    sub_fig=plt.subplot(4,2,3)
    sub_fig.plot(bg[0][ch],c=colors[1])
    sub_fig.set_xlim(0,250)
    sub_fig.set_xticks([0,125,250])
    sub_fig.set_xticklabels([0,0.5,1])
    # sub_fig.set_ylabel('Bg',rotation=0, labelpad=25)
    sub_fig.set_title('Bg',x=-.25,y=.3)
    # sub_fig.set_xlabel('Time(s)')

    sub_fig=plt.subplot(4,2,4)
    freq,spec=FFT(srate,bg[0][ch])
    sub_fig.plot(freq,spec,c=colors[1])
    sub_fig.set_xlim(0,20)
    # sub_fig.set_xlabel('Frequency(Hz)')


    # sub_fig=plt.subplot(5,2,5)
    # sub_fig.plot(template[0][0])

    # sub_fig=plt.subplot(5,2,6)
    # freq,spec=FFT(srate,template[0][0])
    # sub_fig.plot(freq,spec)
    # sub_fig.set_xlim(0,40)
    

    sub_fig=plt.subplot(4,2,5)

    sub_fig.plot(bg_t[0][ch],c=colors[2])
    sub_fig.set_xlim(0,250)
    sub_fig.set_xticks([0,125,250])
    sub_fig.set_xticklabels([0,0.5,1])
    # sub_fig.set_xlabel('Time(s)')
    
    # sub_fig.yaxis.tick_right()
    # sub_fig.set_ylabel(r'$\mu V$',rotation=0,x=5,y=0)
    # sub_fig.set_yticks([])
    sub_fig.set_title('Bg_T',x=-.25,y=.3)
    
    sub_fig=plt.subplot(4,2,6)
    freq,spec=FFT(srate,bg_t[0][ch])
    sub_fig.plot(freq,spec,c=colors[2])
    sub_fig.set_xlim(0,20)
    # sub_fig.set_xlabel('Frequency(Hz)')
    

    sub_fig=plt.subplot(4,2,7)
    sub_fig.plot(gen_trials[0][ch],c=colors[3])
    sub_fig.set_xlim(0,250)
    sub_fig.set_xticks([0,125,250])
    sub_fig.set_xticklabels([0,0.5,1])
    # sub_fig.set_ylabel('Generated\nSample',rotation=0, labelpad=25)
    sub_fig.set_title('Generated\nSample',x=-.25,y=.3)
    sub_fig.set_xlabel('Time(s)')

    sub_fig=plt.subplot(4,2,8)
    freq,spec=FFT(srate,gen_trials[0][ch])
    sub_fig.plot(freq,spec,c=colors[3])
    sub_fig.set_xlim(0,20)
    sub_fig.set_xlabel('Frequency(Hz)')
    

    plt.subplots_adjust(top=0.98,bottom=0.1,left=0.15,right=0.95,hspace=0.5,wspace=0.2)

plt.show()
