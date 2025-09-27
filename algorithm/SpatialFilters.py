from turtle import forward
from sympy import N
import torch
import torch.nn as nn


class Linear_Filter(nn.Module):
    def __init__(self,nchannel):
        super().__init__()
        self.spatial_filter=nn.Linear(nchannel,1)
        self.reverse_filter=nn.Linear(1,nchannel)

    def forward(self,x):
        self.spatial_filter.weight.data.clamp_(min=0.1)
        self.reverse_filter.weight.data.clamp_(min=0.1)

        task_related_component=self.spatial_filter(x)
        template=self.reverse_filter(task_related_component)


        return task_related_component, template
        

class Nonlinear_Filter(nn.Module):
    def __init__(self,nchannel):
        super().__init__()
        self.spatial_filter=nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(nchannel,1)
        )
        self.reverse_filter=nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(1,nchannel)
        )

    def forward(self,x):

        self.spatial_filter[1].weight.data.clamp_(min=0.1)
        self.reverse_filter[1].weight.data.clamp_(min=0.1)

        task_related_component=self.spatial_filter(x)
        template=self.reverse_filter(task_related_component)


        return task_related_component, template


from torch.utils.data import TensorDataset,DataLoader

def gen_pair_data(eeg,batch_size):
    x1_list=[]
    x2_list=[]
    cl_list=[]
    nchannel,nsample,nclass,ntrial=eeg.shape
    for cl in range(nclass):
        for nt in range(ntrial-1):
            for ntt in range(nt+1,ntrial):
                x1=eeg[:,:,cl,nt]
                x2=eeg[:,:,cl,ntt]
                # gen_data.append((x1,x2,cl))
                x1_list.append(x1)
                x2_list.append(x2)
                cl_list.append(cl)

    seg=int(len(x1_list)*0.8)

    train_dataset=TensorDataset(
            torch.as_tensor(x1_list[:seg], dtype=torch.float),
            torch.as_tensor(x2_list[:seg], dtype=torch.float),
            torch.as_tensor(cl_list[:seg], dtype=torch.long)
        )
    train_data_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    
    )

    valid_dataset=TensorDataset(
            torch.as_tensor(x1_list[seg:], dtype=torch.float),
            torch.as_tensor(x2_list[seg:], dtype=torch.float),
            torch.as_tensor(cl_list[seg:], dtype=torch.long)
        )
    valid_data_loader=DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True
    
    )

    return train_data_loader,valid_data_loader
 