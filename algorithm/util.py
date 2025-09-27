# -*- coding: utf-8 -*-
#

import torch
import torch.nn as nn
import numpy as np

class MaxNormConstraintConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm_value=1, norm_axis=2, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True))
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= (desired/norms)
        return w


class MaxNormConstraintLinear(nn.Linear):
    def __init__(self, *args, max_norm_value=1, norm_axis=0, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)
    
    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True))
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= (desired/norms)
        return w  

class SeparableConv2d(nn.Module):
    """An equally SeparableConv2d in Keras.
    A depthwise conv followed by a pointwise conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros', D=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels*D, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False, padding_mode=padding_mode)
        self.pointwise_conv = nn.Conv2d(in_channels*D, out_channels, 1, stride=1, padding=0, bias=bias)
        self.model = nn.Sequential(self.depthwise_conv, self.pointwise_conv)

    def forward(self, X):
        return self.model(X)
    
class MaxNormConstraint:

    def __init__(self,max_value=1,axis=None):
        self.max_value=max_value
        self.axis=axis

    def __call__(self, module):
        if hasattr(module,'weight'):
            weights=module.weight.data
            norms=torch.sqrt(torch.sum(torch.pow(weights,2),dim=self.axis,keepdim=True))
            desired=torch.clamp(norms,min=0,max=self.max_value)
            weights*=desired/(np.finfo(np.float).eps+norms)
            module.weight.data=weights
            return module
        

def compute_conv_outsize(input_size, kernel_size, stride=1, padding=0, dilation=1):
    return int((input_size - dilation * kernel_size + 2 * padding) / stride + 1)
def compute_out_size(input_size: int, kernel_size: int,
        stride: int = 1, padding: int = 0, dilation: int = 1):
    return int((input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)

def compute_same_pad1d(input_size, kernel_size, stride=1, dilation=1):
    all_padding = ((stride-1)*input_size - stride + dilation*(kernel_size-1) + 1)
    return (all_padding//2, all_padding-all_padding//2)

def compute_same_pad2d(input_size, kernel_size, stride=(1, 1), dilation=(1, 1)):
    ud = compute_same_pad1d(input_size[0], kernel_size[0], stride=stride[0], dilation=dilation[0])
    lr = compute_same_pad1d(input_size[1], kernel_size[1], stride=stride[1], dilation=dilation[1])
    return [*lr, *ud]

def generate_tensors(*args, dtype=torch.float):
    new_args = []
    for arg in args:
        new_args.append(torch.as_tensor(arg, dtype=dtype))
    if len(new_args) == 1:
        return new_args[0]
    else:
        return new_args


