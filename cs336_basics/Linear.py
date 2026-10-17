import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device:None,dtype:None):
        super().__init__()
        factory_kwargs={"device":device,"dtype":dtype}
        self.weight=nn.Parameter(torch.empty((out_features,in_features),**factory_kwargs))
        #Xavier思想，std标准差，为了保持层与层之间输出方差的一致性，确保信号能够传到深层。截断范围[-3*std, 3*std](正态分布)
        std=(2.0/(in_features+out_features))**0.5
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=-3*std,b=3*std)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return torch.einsum('...i,oi->...o',x,self.weight)
'''
Linear 层的作用是：用一组可以学习的权重，把输入向量重新组合成一个输出向量。
token ID
   ↓ Embedding
128 维向量
   ↓ Linear(128, 256)
256 维向量
'''
'''
x=[batch,in_features]
w=[out_features,in_features]
y=x* w转置
'''