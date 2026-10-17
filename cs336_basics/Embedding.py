import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int,device=None,dtype=None):
        super().__init__()
        factory_kwargs={'device':device,'dtype':dtype}
        #torch.empty:创建张量，分配空间，不初始化
        self.weight=nn.Parameter(
            torch.empty((num_embeddings,embedding_dim),**factory_kwargs)
            )
        #初始化self.weight数值，用截断正态分布
        nn.init.trunc_normal_(self.weight,mean=0.0,std=1.0,a=-3,b=3)
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.weight[token_ids]

