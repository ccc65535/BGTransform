from pdb import Restart
from re import split
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from algorithm.util import *
from algorithm.EEGNet import EEGNet,EEGNet_dualTask
from augmentation import RsTransfer as rst
from augmentation import BGTransform as bgt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader

import random
from lib.loss_function import *
from lib.LoadSet import *





x_dtype, y_dtype = torch.float, torch.long


durations = [.2,.3,.4,.5]
device_id=0
device = torch.device("cuda:{:d}".format(device_id) if torch.cuda.is_available() else "cpu")
seed=42

# PZ PO5 PO3 POz PO4 PO6 O1 Oz O2
channels = [47, 53, 54, 55, 56, 57, 60, 61, 62]

srate = 250

offLength = 0.14 * srate
start = int((0.5 * srate) + offLength)

rs_gap=int(.5*srate)


root_dir = 'D:\\EEGData\\MetaBCI\\MNE-tsinghua-data\\upload\\yijun\\'
eeg_all=load_SSVEP_Benchmark(root_dir,channels)
np.save('SSVEP.npy',eeg_all)

eeg_all=np.load('./SSVEP.npy')

res_file_name='./record/eegnet-ssvep-bgtransform-'+time.strftime('%m-%d-%H-%M')+'.xlsx'
res_file=pd.ExcelWriter(res_file_name)



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
    for fold,(train_ind,test_ind) in enumerate( kf.split(trials)):
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
        

        # trainX=eeg[:,:,:,train_trials,:]
        # validateX=eeg[:,:,:,valid_trials,:]
        # testX=eeg[:,:,:,test_trials,:]

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
        batch_size,max_epochs,lr = 256,600,1e-3

        train_dataset=TensorDataset(
                trainX,
                trainY,
                trainSub
            )
        train_data_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        
        )

        valid_dataset=TensorDataset(
                validateX,
                validateY,
                validateSub,
        )
        valid_data_loader=DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True
        
        )  




        
        all_model=EEGNet(
                nchannel, int(srate*duration), nclass,
                time_kernel=(96, (1, int(srate*duration)), (1, 1)), 
                D=1,
                separa_kernel=(96, (1, 16), (1, 1)),
                dropout_rate=0.2,
                fc_norm_rate=1).to(device)




        save_path='./model_save/eegnet-fold-'+str(fold)+'-'+str(duration)+'s-bench.pth'

        loss_fun1=nn.CrossEntropyLoss()
        loss_fun2=nn.MSELoss()
        # loss_fun2=corrLoss()
        optimizer=optim.Adam(all_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

        patience=50
        min_loss=np.inf
        last_epoch=0
        a=.2

        for epoch in range(max_epochs):

            # if epoch>50:
            #     patience=5
            all_model.train()
            total_loss=0
            for i, data in enumerate(train_data_loader):

  

                x,label,subject=data
                x,label,subject=x.to(device),label.to(device),subject.to(device)

       
                out=all_model(x)
                
                loss=loss_fun1(out,label)
                
                total_loss+=loss
                
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(sub_model.parameters(), 0.5)
                optimizer.step()

            all_model.eval()
            with torch.no_grad():
                valid_loss=0
                valid_pred_labels=[]
                valid_true_labels=[]
                for i, data in enumerate(valid_data_loader):
                    torch.cuda.empty_cache()
                    # sub_model.eval()

                    # lamda = np.random.beta(8,2)

                    x,label,subject=data
                    x,label,subject=x.to(device),label.to(device),subject.to(device)
                   
                  
                    out=all_model(x)
                    
                    # loss=loss_fun1(out1,label)+a*(loss_fun2(out2,temps))
                    loss=loss_fun1(out,label)
                
                

                    valid_loss+=loss
                    valid_pred_labels+=list(out.argmax(axis=1).detach().cpu().numpy())

                    valid_true_labels+=list(label.detach().cpu().numpy())
                    # valid_true_labels+=list(l[:,0].detach().cpu().numpy())
                    

                valid_acc = balanced_accuracy_score(valid_pred_labels, valid_true_labels)

                all_model.eval()
                output= all_model(testX.to(device))
                pred_labels=output.argmax(axis=1).detach().cpu().numpy()

                true_labels = testY.numpy()
                sub_acc = balanced_accuracy_score(pred_labels, true_labels)

                valid_loss/=len(valid_data_loader)

                pts=f'epoch:{epoch},train loss:{total_loss/(len(train_data_loader)):.3f},valid loss:{valid_loss:.4f},valid acc:{valid_acc:.3f},test acc:{sub_acc:.3f}'
                print(pts)

                if valid_loss<min_loss:
                    last_epoch=epoch
                    min_loss=valid_loss
                
                if epoch-last_epoch>patience:
                    break



        last_save=0
        ## fine-tuning
        sub_accs = []
        for sub_id in subjects:

            print(f'fold{fold+1},sub{sub_id+1}')
            # rest_all=np.concatenate((eeg_all[:,:rs_gap,:,:,:],eeg_all[:,-rs_gap:,:,:,:]),axis=3)

            # rest_ind=np.concatenate((train_trials,valid_trials))
            # sub_resting=np.concatenate((eeg_all[:,:rs_gap,:,rest_ind,sub_id],eeg_all[:,-rs_gap:,:,rest_ind,sub_id]),axis=3)
            # sub_resting=sub_resting.reshape(sub_resting.shape[0],sub_resting.shape[1],-1).transpose(2,0,1)
            
  

            train_sub_ind=torch.where(trainSub==sub_id)
            validate_sub_ind=torch.where(validateSub==sub_id)
            test_sub_ind=torch.where(testSub==sub_id)
            
            
            sub_trainX, sub_trainY = trainX[train_sub_ind], trainY[train_sub_ind]
            sub_validateX, sub_validateY = validateX[validate_sub_ind], validateY[validate_sub_ind]
            sub_testX, sub_testY = testX[test_sub_ind], testY[test_sub_ind]
            

            ex=8

            # used_X=torch.cat((sub_trainX,sub_validateX),dim=0)
            # used_Y=np.concatenate((sub_trainY,sub_validateY),axis=0)
            
            # sub_trainX,sub_trainY=rst.generate_data(used_X.numpy(),used_Y,sub_resting,nt=[2 for i in range(ex)]+[3 for i in range(ex)]+[4 for i in range(ex)])
            # sub_validateX,sub_validateY=used_X.numpy(),used_Y


            aug_X,aug_y=bgt.generate_data(sub_trainX.numpy(),sub_trainY,'ChannelsShuffle',nt=[2 for i in range(int(ex))]+[3 for i in range(ex)])
            sub_trainX=torch.cat((sub_trainX,torch.tensor(aug_X)),dim=0)
            sub_trainY=np.concatenate((sub_trainY,aug_y),axis=0)


            sub_testX=sub_testX.to(device)
            # sub_testY=torch.tensor(sub_testY,dtype=torch.long).to(device)



            batch_size,max_epochs,lr = 128,600,5e-4
            # lamda=0.8
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

            sub_model.load_state_dict(all_model.state_dict())

   


            # loss_fun1=nn.CrossEntropyLoss()
            # loss_fun2=nn.MSELoss()
            optimizer=optim.Adam(sub_model.parameters(), lr=lr,weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

            max_acc=0
            min_loss=np.inf
            save_path='./model_save/bench-eegnet-bgt-'+str(duration)+'s-fold-'+str(fold)+'-sub-'+str(sub_id)+'.pth'

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
                    
                    out=sub_model(x)


                    loss=loss_fun1(out,label)
                    
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

                        out=sub_model(x)



                        # loss=loss_fun1(out1,label)+a*(loss_fun2(out2,temps))
                        loss=loss_fun1(out,label)

                        valid_loss+=loss
                        valid_pred_labels+=list(out.argmax(axis=1).detach().cpu().numpy())

                        # valid_true_labels+=list(l[:,0].detach().cpu().numpy())
                        valid_true_labels+=list(label.detach().cpu().numpy())

                    del x,label
    
                    valid_acc = balanced_accuracy_score(valid_pred_labels, valid_true_labels)

                    sub_model.eval()
                    output = sub_model(sub_testX)
                    pred_labels=output.argmax(axis=1).detach().cpu().numpy()

                    true_labels = sub_testY.numpy()
                    sub_acc = balanced_accuracy_score(pred_labels, true_labels)

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
            sub_acc = balanced_accuracy_score(pred_labels, true_labels)
            sub_accs.append(sub_acc)

        loo_fine_tuning_accs.append(sub_accs)

    global_sub_accs = np.array(loo_global_accs).T
    ft_sub_accs = np.array(loo_fine_tuning_accs).T

    pd.DataFrame(ft_sub_accs).to_excel(excel_writer=res_file,sheet_name=str(duration), index=False, header=False)
    res_file._save()

      