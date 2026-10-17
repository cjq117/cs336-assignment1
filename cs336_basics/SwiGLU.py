import torch
import torch.nn as nn
from cs336_basics.Linear import Linear


#激活函数silu(x)=x*sigmoid(x)
def silu_fn(in_features):
     return in_features*torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device=None,dtype=None):
         super().__init__()
         self.d_ff=d_ff
         self.d_model=d_model
         self.w1=Linear(d_model,d_ff,device,dtype)
         self.w3=Linear(d_model,d_ff,device,dtype)
         self.w2=Linear(d_ff,d_model,device,dtype)

    def forward(self,x:torch.Tensor)->torch.Tensor:
         gate=silu_fn(self.w1(x))
         signal=self.w3(x)
         #这里的*指的是：对应位置的元素相乘
         return self.w2(gate*signal)
         