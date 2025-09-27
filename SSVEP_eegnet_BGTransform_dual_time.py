from pdb import Restart
from re import split
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from augmentation import BGTransform as bgt

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from algorithm.util import *
from algorithm.EEGNet import EEGNet
from augmentation import Augmentation


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader

import random
from lib.loss_function import *
from lib.LoadSet import *





x_dtype, y_dtype = torch.float, torch.long


durations = [0.2,0.3,0.4,0.5]
device_id=0
device = torch.device("cuda:{:d}".format(device_id) if torch.cuda.is_available() else "cpu")
seed=42

# PZ PO5 PO3 POz PO4 PO6 O1 Oz O2
channels = [47, 53, 54, 55, 56, 57, 60, 61, 62]

srate = 250

offLength = 0.14 * srate
start = int((0.5 * srate) + offLength)

rs_gap=int(.5*srate)


# root_dir = 'D:\\EEGData\\MetaBCI\\MNE-tsinghua-data\\upload\\yijun\\'
# eeg_all=load_SSVEP_Benchmark(root_dir,channels)
# np.save('SSVEP.npy',eeg_all)

eeg_all=np.load('./SSVEP.npy')

combinations=[
    ('TimeReverse','SmoothTimeMask'),
    ('FrequencyShift','FTSurrogate'),
    ('TimeReverse','FTSurrogate')
]
print('BGTransform')

for aug_names in combinations:

    for i,ex in enumerate([1,2,4,8,16]):


            for duration in durations:
                

                stop = start + int(srate*duration)

                eeg=eeg_all[:,start:stop,:,:,:]
                eeg-=np.mean(eeg,axis=1,keepdims=True)
                nchannel,nsample,nclass,ntrial,nsubject=eeg.shape
                
                folds=range(ntrial)

                trials=[t for t in range(ntrial)]
                subjects=[i for i in range(nsubject)]



                loo_global_accs = []
                loo_global_model_states = []
                loo_fine_tuning_accs = []

                kf=KFold(n_splits=ntrial)

                start_time=time.time()

                for fold,(train_ind,test_ind) in enumerate( kf.split(trials)):
                    if fold>0:
                        break
                    test_trials = test_ind
                    valid_trials =train_ind[-1:]
                    train_trials = train_ind[:-1]

                    random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                    # torch.cuda.manual_seed_all(seed)
                        torch.cuda.manual_seed(seed)
                        # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware.
                        torch.backends.cudnn.benchmark = False
                        # Certain operations in Cudnn are not deterministic, and this line will force them to behave!
                        torch.backends.cudnn.deterministic = True
                    np.random.seed(seed)
                    

                    trainX,trainY,trainSub=[],[],[]
                    validateX,validateY,validateSub=[],[],[]
                    testX,testY,testSub=[],[],[]

                    for cl in range(nclass):
                        for t in trials:
                            for s in subjects:

                                temp_eeg=eeg[:,:,cl,t,s].squeeze()
                                
                                if t in train_trials:
                                    trainX.append(temp_eeg)
                                    trainY.append(cl)
                                    trainSub.append(s)

                                elif t in valid_trials:
                                    validateX.append(temp_eeg)
                                    validateY.append(cl)
                                    validateSub.append(s)

                                else:
                                    testX.append(temp_eeg)
                                    testY.append(cl)
                                    testSub.append(s)



                    trainX, validateX, testX = generate_tensors(
                        trainX, validateX, testX, dtype=x_dtype)
                    trainY, validateY, testY = generate_tensors(
                        trainY, validateY, testY, dtype=y_dtype)
                    trainSub, validateSub, testSub = generate_tensors(
                        trainSub, validateSub, testSub, dtype=y_dtype)

                    ######
                    

                    all_model_save_path='./model_save/eegnet-fold-'+str(fold)+'-'+str(duration)+'s-bench.pth'

                    sub_accs = []
                    for sub_id in subjects:


                        train_sub_ind=torch.where(trainSub==sub_id)
                        validate_sub_ind=torch.where(validateSub==sub_id)
                        test_sub_ind=torch.where(testSub==sub_id)
                        
                        
                        sub_trainX, sub_trainY = trainX[train_sub_ind], trainY[train_sub_ind]
                        sub_validateX, sub_validateY = validateX[validate_sub_ind], validateY[validate_sub_ind]
                        sub_testX, sub_testY = testX[test_sub_ind], testY[test_sub_ind]


                        aug_X1,aug_y1=bgt.generate_data(sub_trainX.numpy(),sub_trainY,aug_names[0],nt=[2 for i in range(int(ex))])
                        aug_X2,aug_y2=bgt.generate_data(sub_trainX.numpy(),sub_trainY,aug_names[1],nt=[2 for i in range(int(ex))])
                        
                        sub_trainX=torch.cat((sub_trainX,torch.tensor(aug_X1),torch.tensor(aug_X2)),dim=0)
                        sub_trainY=np.concatenate((sub_trainY,aug_y1,aug_y2),axis=0)


                        sub_testX=sub_testX.to(device)



                        batch_size,max_epochs,lr = 128,20,5e-4
                        patience=5

                        sub_train_dataset=TensorDataset(
                                torch.tensor(sub_trainX,dtype=x_dtype),
                                torch.tensor(sub_trainY,dtype=torch.long),
                                
                        )
                        sub_train_data_loader=DataLoader(
                            sub_train_dataset,
                            batch_size=batch_size,
                            shuffle=True
                        
                        )            
                        
                        sub_valid_dataset=TensorDataset(
                            torch.tensor(sub_validateX,dtype=x_dtype),
                            torch.tensor(sub_validateY,dtype=torch.long),
                        )
                        sub_valid_data_loader=DataLoader(
                            sub_valid_dataset,
                            batch_size=batch_size,
                            shuffle=True
                        
                        )  
                        

                        sub_model=EEGNet(
                            nchannel, int(srate*duration), nclass,
                            time_kernel=(96, (1, int(srate*duration)), (1, 1)), 
                            D=1,
                            separa_kernel=(96, (1, 16), (1, 1)),
                            dropout_rate=0.2,
                            fc_norm_rate=1
                            ).to(device)

                        sub_model.load_state_dict(torch.load(all_model_save_path))

                        loss_fun1=nn.CrossEntropyLoss()
                        # loss_fun2=nn.MSELoss()
                        optimizer=optim.Adam(sub_model.parameters(), lr=lr,weight_decay=1e-3)
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

                        max_acc=0
                        min_loss=np.inf
                        save_path='./model_save/bench-eegnet-aug-'+str(duration)+'s-fold-'+str(fold)+'-sub-'+str(sub_id)+'.pth'

                        for epoch in range(max_epochs):

                          
                            sub_model.train()
                            total_loss=0
                            for i, data in enumerate(sub_train_data_loader):

                                # lamda = np.random.beta(8,2)

                                x,label=data
                                x,label=x.to(device),label.to(device)
                                # sub_model.train()
                                
                                out=sub_model(x)


                                loss=loss_fun1(out,label)
                                
                                total_loss+=loss
                                
                                optimizer.zero_grad()
                                loss.backward()
                                # torch.nn.utils.clip_grad_norm_(sub_model.parameters(), 0.5)
                                optimizer.step()

                        

                end_time=time.time()

                train_time = (end_time - start_time)

                print(f'{aug_names[0]}-{aug_names[1]}-{duration}s-{ex*2}times:{train_time:.4f}')