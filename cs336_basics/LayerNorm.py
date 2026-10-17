import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        factory_kwargs = {'device':device,'dtype':dtype}
        #可学习的参数γ，β
        self.weight=nn.Parameter(torch.ones(d_model,**factory_kwargs))
        self.bias=nn.Parameter(torch.zeros(d_model,**factory_kwargs))
        #eps：很小的数，防止除以零
        self.eps=eps

    def forward(self,x:torch.Tensor)->torch.Tensor:
        #获取x的类型，到后面再转换回来
        in_dtype=x.dtype
        #转换成float32位，确保计算均值和方差时的数值稳定性
        x_float=x.to(torch.float32)
        #dim=-1在最后一维算，也就是D[B,S,D]
        #keepdim=True保留维度
        mean=x_float.mean(dim=-1,keepdim=True)
        #unbiased=False:LayerNorm 的方差应除以特征数D,而不是D-1。
        var=x_float.var(dim=-1,keepdim=True,unbiased=False)
        x_norm=(x_float-mean)/torch.sqrt(var+self.eps)
        result=x_norm*self.weight+self.bias
        return result.to(in_dtype)


