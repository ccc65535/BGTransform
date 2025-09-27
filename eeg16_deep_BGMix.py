from heapq import nsmallest
from re import sub
import time
import warnings
import winsound
from matplotlib.pyplot import axis
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mne
import joblib
import scipy
import os,copy


import sys
sys.path.append(r"D:\\Projects\\Python\\metabci")



from sklearn.metrics import confusion_matrix, balanced_accuracy_score,accuracy_score


from algorithm.deepnet import Deep4Net

from algorithm.util import *


import torch, skorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skorch.classifier import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import (LRScheduler, EpochScoring, Checkpoint, Callback,
                              TrainEndCheckpoint, LoadInitState, EarlyStopping)

from torch.utils.data import TensorDataset,DataLoader
from augmentation import BGTransform as bgt

device_id = 0
device = torch.device("cuda:{:d}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print("Total available GPU devices: {}".format(torch.cuda.device_count()))
print("Current pytorch device: {}".format(device))


x_dtype, y_dtype = torch.float, torch.long

model_name = 'deepnet'



SEED=42

#########################


build_model_ch = [i for i in range(8)]
template_ch = build_model_ch
ch_names=['PO5','PO3','POz','PO4','PO6','O1','Oz','O2']


#########
srate = 500
calTime = 4

offLength = 0.14 * srate
sampleCount = int(calTime * srate)
start = int((2 * srate) + offLength)
end = start + sampleCount



########
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#################

aug_names=['TimeReverse','FTSurrogate']
ex=8

nSubs=10
nclass=16
nTrial=6


subs = [s for s in range(nSubs)]
labels = [i for i in range(nclass)]
trials = [t for t in range(nTrial)]

eeg_all = np.load('D:/EEGData/eeg16.npy').squeeze()


n_classes = len(labels)

# durations = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
durations = [.2,.3,.4,.5]


res_file_name='./record/eeg16-deep-'+aug_names[0]+'-'+aug_names[1]+'-'+str(ex*4)+'times.xlsx'
res_file=pd.ExcelWriter(res_file_name)


for duration in durations:
    
    eeg=eeg_all[:,start:start+int(duration*srate),:,:]
    sub_ids=np.linspace(0,nSubs-1,nSubs)
    sub_ids=np.expand_dims(sub_ids,0).repeat(nTrial,axis=0)

    loo_global_model_states=[]
    loo_fine_tuning_accs=[]
        ###
    for fold in trials:
        test_trial = [fold]
        valid_trials = [(fold + 1) % 6]
        train_trials = list(set(trials).difference(valid_trials+test_trial ))

        trainX, validateX, testX = [],[],[]
        trainY, validateY, testY =[],[],[]
        trainSub,validateSub,testSub=[],[],[]

        for l in labels:
            for s in subs:
                for t in trials:
                    if t in train_trials:
                        trainX.append(eeg[:,:,l,t,s])
                        trainY.append(l)
                        trainSub.append(s)
                    elif t in valid_trials:
                        validateX.append(eeg[:,:,l,t,s])
                        validateY.append(l)
                        validateSub.append(s)
                    else:
                        testX.append(eeg[:,:,l,t,s])
                        testY.append(l)
                        testSub.append(s)
            

        trainX, validateX, testX=np.array(trainX), np.array(validateX), np.array(testX)
        trainY, validateY, testY =np.array(trainY), np.array(validateY), np.array(testY)
        trainSub,validateSub,testSub=np.array(trainSub), np.array(validateSub), np.array(testSub)


        trainX, validateX, testX = generate_tensors(
            trainX, validateX, testX, dtype=x_dtype)
        trainY, validateY, testY = generate_tensors(
            trainY, validateY, testY, dtype=y_dtype)
       
        ######

        n_samples=int(srate*duration)

        batch_size,max_epochs,lr = 256,600,1e-3

        _, nchannel, nsample = trainX.shape

        all_model=Deep4Net(
                        n_chans=nchannel, 
                        input_window_samples=nsample,
                        n_classes=nclass,
                        n_filters_time=40,
                        filter_time_length=15,
                        n_filters_spat=40,
                        filter_length_2=5,
                        pool_time_length=5,
                        pool_time_stride=1,
                        drop_prob=0.2).to(device=device)
        
        
        net=NeuralNetClassifier(
            module=all_model,
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            optimizer__weight_decay=0,
            lr=lr,
            max_epochs = max_epochs,
            batch_size = batch_size,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            device=device,

            callbacks=[
                    ('train_acc', EpochScoring('accuracy', 
                                                name='train_acc', 
                                                on_train=True, 
                                                lower_is_better=False)),
                    ('lr_scheduler', LRScheduler('CosineAnnealingLR', T_max=300 - 1)),
                    ('estoper', EarlyStopping(patience=50)),
                    ('checkpoint', Checkpoint(dirname="checkpoints/{:s}".format(str(id(all_model))))),
            ],  

        )


        net.train_split = predefined_split(
                skorch.dataset.Dataset(
                    X=validateX.to(device),
                    y=validateY.to(device)
                    )
        )
            
        save_path='./model_save/deep-fold-'+str(fold)+'-'+str(duration)+'s-eeg16.pth'
        tunning=True
        new_test=True
        # new_test = False
        if tunning:
            if new_test:
                net = net.fit(
                    X= trainX.to(device), 
                    y=trainY.to(device)
                    )
                torch.save(all_model.state_dict(),save_path)
            else:
                net.initialize()
                all_model.load_state_dict(torch.load(save_path))
                print('load all sub model')

        else:
            net = net.fit(
                    X= trainX.to(device), 
                    y=trainY.to(device)
            )


        loo_global_model_states.append(
            copy.deepcopy(net.module.state_dict()))

  



        last_save=0
        ## fine-tuning
        sub_accs = []
        for sub_id in subs:
        # for sub_id in [2,9,10]:
            print(f'fold{fold},sub{sub_id}')
            sub_train_ind = np.where(trainSub==sub_id)
            sub_valid_ind = np.where(validateSub==sub_id)
            sub_test_ind = np.where(testSub==sub_id)
            
            sub_trainX, sub_trainY = trainX[sub_train_ind], trainY[sub_train_ind]
            sub_validateX, sub_validateY = validateX[sub_valid_ind], validateY[sub_valid_ind]
            sub_testX, sub_testY = testX[sub_test_ind], testY[sub_test_ind]


            
            
            ###
            used_X=torch.cat((sub_trainX,sub_validateX),dim=0)
            used_Y=np.concatenate((sub_trainY,sub_validateY),axis=0)
            used_Y_zip=np.array(list(zip(used_Y,used_Y)))
            

            aug_X1,aug_y1=bgt.generate_data(sub_trainX.numpy(),sub_trainY,aug_names[0],nt=[2 for i in range(int(ex))]+[3 for i in range(ex)])
            aug_X2,aug_y2=bgt.generate_data(sub_trainX.numpy(),sub_trainY,aug_names[1],nt=[2 for i in range(int(ex))]+[3 for i in range(ex)])
            
            sub_trainX=torch.cat((sub_trainX,torch.tensor(aug_X1),torch.tensor(aug_X2)),dim=0)
            sub_trainY=np.concatenate((sub_trainY,aug_y1,aug_y2),axis=0)
           
            sub_testX=sub_testX.to(device)
            # sub_testY=torch.tensor(sub_testY,dtype=torch.long).to(device)




            batch_size,max_epochs,lr = 256,600,1e-4
            # lamda=0.8
            patience=5

            train_dataset=TensorDataset(
                torch.tensor(sub_trainX,dtype=x_dtype),
                torch.tensor(sub_trainY,dtype=torch.long),
                
            )
            sub_train_data_loader=DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            
            )  

                        
            valid_dataset=TensorDataset(
                torch.tensor(sub_validateX,dtype=x_dtype),
                torch.tensor(sub_validateY,dtype=torch.long),
            )
            sub_valid_data_loader=DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            sub_model=Deep4Net(
                    n_chans=nchannel, 
                    input_window_samples=nsample,
                    n_classes=nclass,
                    n_filters_time=40,
                    filter_time_length=15,
                    n_filters_spat=40,
                    filter_length_2=5,
                    pool_time_length=5,
                    pool_time_stride=1,
                    drop_prob=0.5).to(device)






            sub_model.load_state_dict(
                copy.deepcopy(loo_global_model_states[fold]))

            loss_fun=nn.CrossEntropyLoss()
            optimizer=optim.Adam(sub_model.parameters(), lr=lr,weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)

            max_acc=0
            min_loss=np.inf
            save_path='./model_save/eeg16-deep-'+str(duration)+'s-fold-'+str(fold)+'-sub-'+str(sub_id)+'.pth'

            for epoch in range(max_epochs):

                # if epoch>50:
                #     patience=5
                sub_model.train()
                total_loss=0
                for i, data in enumerate(sub_train_data_loader):

                    # lamda = np.random.beta(8,2)

                    x,label=data
                    x,label=x.to(device),label.to(device)
                    # sub_model.train()
                    
                    out1=sub_model(x)

                    loss=loss_fun(out1,label)
                    
                    total_loss+=loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(sub_model.parameters(), 0.5)
                    optimizer.step()

                sub_model.eval()
                with torch.no_grad():
                    valid_loss=0
                    valid_pred_labels=[]
                    valid_true_labels=[]
                    for i, data in enumerate(sub_valid_data_loader):
                        torch.cuda.empty_cache()
                        # sub_model.eval()

                        x,label=data
                        x,label=x.to(device),label.to(device)

                        out1=sub_model(x)

                        loss=loss_fun(out1,label)

                        valid_loss+=loss
                        valid_pred_labels+=list(out1.argmax(axis=1).detach().cpu().numpy())

                        # valid_true_labels+=list(l[:,0].detach().cpu().numpy())
                        valid_true_labels+=list(label.detach().cpu().numpy())

                    del x,label
    
                    valid_acc = balanced_accuracy_score(valid_true_labels, valid_pred_labels)

                    sub_model.eval()
                    output = sub_model(sub_testX)
                    pred_labels=output.argmax(axis=1).detach().cpu().numpy()

                    true_labels = sub_testY.numpy()
                    sub_acc = balanced_accuracy_score(true_labels, pred_labels)

                    valid_loss/=len(sub_valid_data_loader)

                    print(f'epoch:{epoch},train loss:{total_loss/(len(sub_train_data_loader)):.3f},valid loss:{valid_loss:.4f},valid acc:{valid_acc:.3f},test acc:{sub_acc:.3f}')

                    # if (valid_loss+5e-4<min_loss):
                    if (valid_loss+1e-4<min_loss) or(valid_acc>max_acc):
                        torch.save(sub_model.state_dict(),save_path)

                        print('save model.')
                        if valid_acc>max_acc:
                            max_acc=valid_acc
                        if valid_loss+1e-4<min_loss:
                            min_loss=valid_loss
                        last_save=epoch
                        

                    if last_save+patience<epoch:
                        print('early stop')
                        break

            state_dict = torch.load(save_path, map_location=device)
            sub_model.load_state_dict(state_dict)
            sub_model.eval()

            output = sub_model(sub_testX)
            pred_labels=output.argmax(axis=1).detach().cpu().numpy()

            true_labels = sub_testY.numpy()
            sub_acc = balanced_accuracy_score(true_labels, pred_labels)
            sub_accs.append(sub_acc)

        loo_fine_tuning_accs.append(sub_accs)


    ft_sub_accs = np.array(loo_fine_tuning_accs).T

    pd.DataFrame(ft_sub_accs).to_excel(excel_writer=res_file,sheet_name=str(duration), index=False, header=False)
    res_file._save()

