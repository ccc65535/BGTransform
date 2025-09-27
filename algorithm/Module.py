
import math
import torch.nn as nn
from torch import Tensor
from algorithm.util import *

class Task_Classifier(nn.Module):
    
    def __init__(
                self,
                embedding_size,
                n_classes,
                norm_rate
                ) -> None:
        super().__init__()

        self.fc_layer = MaxNormConstraintLinear(embedding_size, n_classes, max_norm_value=norm_rate)
        
    def forward(self, emb_X):
        return self.fc_layer(emb_X)



class Task_Regression(nn.Module):
    def __init__(
                self,
                in_channel,
                out_channel,
                in_size,
                out_size,
                stride=(2,2)
                ) -> None:
        super().__init__()

        k0=out_size[0]-(in_size[0]-1)*stride[0]
        k1=out_size[1]-(in_size[1]-1)*stride[1]

        self.Deconv = nn.ConvTranspose2d(in_channel,out_channel,(k0,k1),stride)
        
    def forward(self, middle_X):
        return self.Deconv(middle_X).squeeze()
    

class Task_Regression_Seq(nn.Module):
    def __init__(
                self,
                in_channel,
                out_channel,
                in_size,
                out_size,
                stride=(2,2)
                ) -> None:
        super().__init__()

        k0=out_size[0]-(in_size[0]-1)*stride[0]
        k1=out_size[1]-(in_size[1]-1)*stride[1]

        hid_channel,n_head,dim_feedforward=out_channel,2,out_channel

        self.Deconv = nn.ConvTranspose2d(in_channel,hid_channel,(k0,k1),stride)

        TF_layer=nn.TransformerEncoderLayer(hid_channel,n_head,dim_feedforward,batch_first=True)

        self.TF=nn.TransformerEncoder(TF_layer,num_layers=2)

        self.pos_encoder = PositionalEncoding(d_model=hid_channel)

        # self.fc=nn.Linear(hid_channel,out_channel)
        
    def forward(self, middle_X):

        deOut=self.Deconv(middle_X).squeeze().transpose(1,2)

        posOut = self.pos_encoder(deOut)
        tfOut=self.TF(posOut)
        # out=self.fc(tfOut)

        return tfOut.transpose(1,2)
    

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x=x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.transpose(0,1))
    

