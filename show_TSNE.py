import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.manifold import TSNE 

from augmentation import Augmentation
from augmentation import BGTransform as bgt

from lib.LoadSet import *

from sklearn.model_selection import KFold
import torch
from algorithm.EEGNet import EEGNet



def tsne(X,y,X_,y_):
    X=X.reshape(X.shape[0],-1)
    X_=X_.reshape(X_.shape[0],-1)

    # y=[int(i) for i in y]

    # X_std = StandardScaler().fit_transform(X) 
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # X_tsne = tsne.fit_transform(X_std) 
    # X_tsne_data = np.vstack((X_tsne.T, y)).T 
    # df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class']) 
    # df_tsne.head()

    plt.figure(figsize=(6, 6)) 

    color_map = plt.get_cmap('jet',len(np.unique(y)))
    colors_y=color_map(y)
    colors_y_=color_map(y_)

    X_std = StandardScaler().fit_transform(X) 
    X_std_ = StandardScaler().fit_transform(X_) 

    n_x=X.shape[0]

    X_all=np.concatenate((X_std,X_std_),axis=0)
    
    tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=25,init='pca', n_iter=2000)
    X_trans = tsne.fit_transform(X_all)

    plt.scatter(X_trans[n_x:, 0], X_trans[n_x:, 1], c=colors_y_, cmap=plt.cm.Spectral,marker='+')
    plt.scatter(X_trans[:n_x, 0], X_trans[:n_x, 1], c=colors_y, cmap=plt.cm.Spectral,marker='o',edgecolors='black')
    

    # plt.legend()
    plt.xticks([],[])
    plt.yticks([],[])


def tsne_(X,y,X_,y_,X_d,y_d):
    X=X.reshape(X.shape[0],-1)
    X_=X_.reshape(X_.shape[0],-1)
    X_d=X_.reshape(X_d.shape[0],-1)



    plt.figure(figsize=(6, 6)) 

    color_map = plt.get_cmap('jet',len(np.unique(y)))
    colors_y=color_map(y)
    colors_y_=color_map(y_)
    colors_y_d=color_map(y_d)

    X_std = StandardScaler().fit_transform(X) 
    X_std_ = StandardScaler().fit_transform(X_) 
    X_std_d = StandardScaler().fit_transform(X_d) 

    n_x=X.shape[0]
    n_x_d=X_d.shape[0]

    X_all=np.concatenate((X_std,X_std_,X_std_d),axis=0)
    
    tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=25,init='pca', n_iter=2000)
    X_trans = tsne.fit_transform(X_all)


    subp1=plt.subplot(1,2,1)
    subp1.scatter(X_trans[:n_x, 0], X_trans[:n_x, 1], c=colors_y, cmap=plt.cm.Spectral,marker='o',edgecolors='black')
    subp1.scatter(X_trans[n_x:-n_x_d, 0], X_trans[n_x:-n_x_d, 1], c=colors_y_, cmap=plt.cm.Spectral,marker='+')
    plt.xticks([],[])
    plt.yticks([],[])

    subp2=plt.subplot(1,2,2)
    subp2.scatter(X_trans[:n_x, 0], X_trans[:n_x, 1], c=colors_y, cmap=plt.cm.Spectral,marker='o',edgecolors='black')
    subp2.scatter(X_trans[-n_x_d:, 0], X_trans[-n_x_d:, 1], c=colors_y_d, cmap=plt.cm.Spectral,marker='x')
    plt.xticks([],[])
    plt.yticks([],[])



def model(sub_trainX,save_path): 

    device='cuda:0'
    sub_model=EEGNet(
                nchannel, int(srate*duration), nclass,
                time_kernel=(96, (1, int(srate*duration)), (1, 1)), 
                D=1,
                separa_kernel=(96, (1, 16), (1, 1)),
                dropout_rate=0.2,
                fc_norm_rate=1
                ).to(device)

    state_dict = torch.load(save_path, map_location=device)
    sub_model.load_state_dict(state_dict)
    sub_model.eval()

    sub_trainX=torch.tensor(sub_trainX,dtype=torch.float)
    embedding= torch.nn.Sequential(sub_model.step1, sub_model.step2, sub_model.step3)
    embedding_out=embedding(sub_trainX.to(device).unsqueeze(1)).detach().cpu().numpy()

    return embedding_out




srate = 250
root_dir = 'D:\\EEGData\\MetaBCI\\MNE-nakanishi2015-data\\mnakanishi\\12JFPM_SSVEP\\raw\\master\\data\\'
eeg_all=load_Nak(root_dir,srate)
eeg_all


srate = 250

offLength = 0.135 
start =int(srate*offLength)

duration=0.5
fold=0
#1,3,4,5,7,9
sub_id=2
    

stop = start + int(srate*duration)

eeg=eeg_all[:,start:stop,:,:,:]
eeg-=np.mean(eeg,axis=1,keepdims=True)
nchannel,nsample,nclass,ntrial,nsubject=eeg.shape
folds=range(ntrial)
trials=[t for t in range(ntrial)]


n_splits=15


trian_trials=list(set(trials).difference([fold]))

sub_data=eeg[:,:,:,trian_trials,sub_id]

nch,nsamp,nclass,ntrail=sub_data.shape


sub_trainX,sub_trainY=[],[]
for c in range(nclass):
    for t in range(ntrail):
        sub_trainX.append(sub_data[:,:,c,t])
        sub_trainY.append(c)
sub_trainX=np.array(sub_trainX)
sub_trainY=np.array(sub_trainY)


aug_names=['TimeReverse','SmoothTimeMask','FrequencyShift','FTSurrogate']

ex=16
for type in aug_names:
    aug_X,aug_y=bgt.generate_data(sub_trainX,sub_trainY,type,nt=[2 for i in range(int(ex))]+[3 for i in range(ex)])


    aug_X_d,aug_y_d=Augmentation.augment(type,ex*2,sub_trainX,sub_trainY,srate)

    save_path='./model_save/nak-eegnet-bgt-'+str(duration)+'s-fold-'+str(fold)+'-sub-'+str(sub_id)+'.pth'


    sub_trainX_p=model(sub_trainX,save_path)
    aug_X_p=model(aug_X,save_path)
    tsne(sub_trainX_p,sub_trainY,aug_X_p,aug_y)
    plt.title(type)


    # aug_X_d_p=model(aug_X_d,save_path)
    # tsne_(sub_trainX_p,sub_trainY,aug_X_p,aug_y,aug_X_d_p,aug_y_d)

plt.show()
