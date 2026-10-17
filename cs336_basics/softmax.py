import torch
#在transformer自注意力里面，softmax通常dim=-1（在最后一维归一化）
def softmax(x:torch.Tensor,dim:int =-1)->torch.Tensor:
    #keepdim=True保留维度。
    #x_max沿着指定的维度取最大值
    #只在 dim=k 内部取一组数计算 softmax；不同组之间互不影响
    x_max=torch.max(x,dim=dim,keepdim=True).values
    #该维度的该组，所有数减去最大值，保证数值稳定。但是结果不变，因为指数运算减法可以拆成除法
    x_stable=x-x_max
    exp_x=torch.exp(x_stable)
    sum_exp=torch.sum(exp_x,dim=dim,keepdim=True)
    return exp_x/sum_exp


