from json import load
from re import split
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from algorithm.util import *
from algorithm.EEGNet import EEGNet


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader

import random
from lib.loss_function import *
from lib.LoadSet import *
import pandas as pd
import copy
from augmentation import BGTransform as bgt


def inba_k_fold(k,kfold,meta):

    meta_new=copy.deepcopy(meta)
    meta_new.insert(2,'dataset','None')

    for sub in range(1,26):

        non_tar_ind=meta[(meta['subject']==sub) &( meta['class']==0)].index
        tar_ind=meta[(meta['subject']==sub) &( meta['class']==1)].index

        ##
        seg0=int(len(non_tar_ind)/kfold)
        st,ed=int(k*seg0),int((k+1)*seg0)

        test_ind=non_tar_ind[st:ed]
        rest_ind=[i for i in non_tar_ind if i not in test_ind]
        train_ind=rest_ind[:-seg0]
        valid_ind=rest_ind[-seg0:]
        
        meta_new.loc[train_ind,'dataset']='train'
        meta_new.loc[valid_ind,'dataset']='valid'
        meta_new.loc[test_ind,'dataset']='test'

        ##
        seg1=int(len(tar_ind)/kfold)
        st,ed=int(k*seg1),int((k+1)*seg1)

        test_ind=tar_ind[st:ed]
        rest_ind=[i for i in tar_ind if i not in test_ind]
        train_ind=rest_ind[:-seg1]
        valid_ind=rest_ind[-seg1:]

        meta_new.loc[train_ind,'dataset']='train'
        meta_new.loc[valid_ind,'dataset']='valid'
        meta_new.loc[test_ind,'dataset']='test'

    return meta_new




    

x_dtype, y_dtype = torch.float, torch.long


durations = [.5]
device_id=0
device = torch.device("cuda:{:d}".format(device_id) if torch.cuda.is_available() else "cpu")
seed=42

subjects=range(1,26)
srate = 100


# eeg_all,meta=load_BrainInv()
# np.save('./P3_data.npy',eeg_all)
# meta.to_csv('P3_meta.csv',index=False)

eeg_all=np.load('./P3_data.npy')
meta_all=pd.read_csv('P3_meta.csv')
y=meta_all['class'].values
subs=meta_all['subject'].values


# res_file_name='./record/eegnet-dualtask-P3-'+time.strftime('%m-%d-%H-%M')+'.xlsx'
# res_file=pd.ExcelWriter(res_file_name)

nclass=2
aug_names=['TimeReverse','SmoothTimeMask','FrequencyShift','FTSurrogate']

first_train=True
for aug_name in aug_names:

    for i,ex in enumerate([1,2,4,8,16]):
        if i!=0:
            first_train=False

        res_file_name='./record/P3-eegnet-'+aug_name+'-'+str(ex*2)+'times.xlsx'
        res_file=pd.ExcelWriter(res_file_name)
        # for duration in durations:

        duration=0.5
        
        t=int(srate*duration)


        eeg=eeg_all[:,:,:t]

        ntrial,nchannel,nsample=eeg.shape
    



        loo_global_accs = []
        loo_global_model_states = []
        loo_fine_tuning_accs = []

        kfold=5

        for k in range(kfold):
            meta=inba_k_fold(k,kfold,meta_all)

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
            

            
            train_ind=meta.loc[meta['dataset']=='train'].index
            valid_ind=meta.loc[meta['dataset']=='valid'].index
            test_ind=meta.loc[meta['dataset']=='test'].index

            
            trainX, validateX, testX=eeg[train_ind],eeg[valid_ind],eeg[test_ind]
            trainY, validateY, testY=y[train_ind],y[valid_ind],y[test_ind]
            trainSub, validateSub, testSub =subs[train_ind],subs[valid_ind],subs[test_ind]
                

            

    


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



            all_model_save_path='./model_save/eegnet-fold-'+str(k)+'-'+str(duration)+'s-P3.pth'
            if first_train:
            
                all_model=EEGNet(
                        nchannel, int(srate*duration), nclass,
                        time_kernel=(8, (1, int(srate*duration)), (1, 1)), 
                        D=2,
                        separa_kernel=(16, (1, 16), (1, 1)),
                        dropout_rate=0.2,
                        fc_norm_rate=1).to(device)


                loss_fun=nn.CrossEntropyLoss()
                optimizer=optim.Adam(all_model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

                patience=10
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
                        
                        loss=loss_fun(out,label)
                        
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
                            
                            loss=loss_fun(out,label)
                        
                        

                            valid_loss+=loss
                            valid_pred_labels+=list(out.argmax(axis=1).detach().cpu().numpy())

                            valid_true_labels+=list(label.detach().cpu().numpy())
                            

                        valid_acc = balanced_accuracy_score(valid_pred_labels, valid_true_labels)

                        all_model.eval()
                        output = all_model(testX.to(device))
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
                            torch.save(all_model.state_dict(),all_model_save_path)
                            break



            last_save=0
            ## fine-tuning
            sub_accs = []
            for sub_id in subjects:

                print(f'fold{k},sub{sub_id}')

                train_sub_ind=meta.loc[(meta['subject']==sub_id) &(meta['dataset']=='train')].index
                validate_sub_ind=meta.loc[(meta['subject']==sub_id) &(meta['dataset']=='valid')].index
                test_sub_ind=meta.loc[(meta['subject']==sub_id) &(meta['dataset']=='test')].index
                
                
                sub_trainX, sub_trainY = eeg[train_sub_ind], y[train_sub_ind]
                sub_validateX, sub_validateY = eeg[validate_sub_ind], y[validate_sub_ind]
                sub_testX, sub_testY = eeg[test_sub_ind], y[test_sub_ind]


                aug_X,aug_y=bgt.generate_data(sub_trainX,sub_trainY,aug_name,nt=[2 for i in range(ex)]+[3 for i in range(ex)],srate=srate)
                sub_trainX=torch.cat((torch.tensor(sub_trainX),torch.tensor(aug_X)),dim=0)
                sub_trainY=np.concatenate((sub_trainY,aug_y),axis=0)
                
                sub_testX=torch.tensor(sub_testX,dtype=x_dtype).to(device)
                # sub_testY=torch.tensor(sub_testY,dtype=torch.long).to(device)



                batch_size,max_epochs,lr = 128,600,5e-4
                # lamda=0.8
                patience=10

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
                    time_kernel=(8, (1, int(srate*duration)), (1, 1)), 
                    D=2,
                    separa_kernel=(16, (1, 16), (1, 1)),
                    dropout_rate=0.2,
                    fc_norm_rate=1).to(device)

                sub_model.load_state_dict(torch.load(all_model_save_path))

    
                optimizer=optim.Adam(sub_model.parameters(), lr=lr,weight_decay=1e-3)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

                max_acc=0
                min_loss=np.inf
                save_path='./model_save/p3-eegnet-'+str(duration)+'s-fold-'+str(k)+'-sub-'+str(sub_id)+'.pth'

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




                        loss=loss_fun(out,label)
                        
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

                            loss=loss_fun(out,label)

                            valid_loss+=loss
                            valid_pred_labels+=list(out.argmax(axis=1).detach().cpu().numpy())

                            valid_true_labels+=list(label.detach().cpu().numpy())

                        del x,label
        
                        valid_acc = balanced_accuracy_score(valid_pred_labels, valid_true_labels)

                        sub_model.eval()
                        output = sub_model(sub_testX)
                        pred_labels=output.argmax(axis=1).detach().cpu().numpy()

                        true_labels = sub_testY
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
                            torch.save(sub_model.state_dict(),save_path)
                            break

                state_dict = torch.load(save_path, map_location=device)
                sub_model.load_state_dict(state_dict)
                sub_model.eval()

                output = sub_model(sub_testX)
                pred_labels=output.argmax(axis=1).detach().cpu().numpy()

                true_labels = sub_testY
                sub_acc = balanced_accuracy_score(pred_labels, true_labels)
                sub_accs.append(sub_acc)

            loo_fine_tuning_accs.append(sub_accs)

        global_sub_accs = np.array(loo_global_accs).T
        ft_sub_accs = np.array(loo_fine_tuning_accs).T

        pd.DataFrame(ft_sub_accs).to_excel(excel_writer=res_file,sheet_name=str(duration), index=False, header=False)
        res_file._save()

      