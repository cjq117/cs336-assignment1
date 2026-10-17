import torch

def cross_entropy(logits:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
    m=torch.max(logits,dim=-1,keepdim=True).values
    #使用gather函数从词表维度根据targets提取对应的分数
    #gather要求index的维度和输入一致，需要将targets升维
    #取完之后再降维（squeeze(-1)）
    target_logits=torch.gather(logits,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)
    shifted_logits=logits-m
    #公式：M+log(sum(exp(o-M)))
    log_sum_exp=m.squeeze(-1)+torch.log(torch.sum(torch.exp(shifted_logits),dim=-1))
    loss=log_sum_exp-target_logits
    #对整个批次求平均，返回一个标量
    return torch.mean(loss)