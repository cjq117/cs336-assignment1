import torch
import math
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor=None
)->torch.Tensor:
    '''
    Q:[...,n,d_k]
    K:[...,m,d_k]
    V:[...,m,d_v]
    mask:[n,m],True为保留,false为屏蔽
    '''
    #Q的最后一维给d_k
    d_k=Q.size(-1)
    scores=torch.einsum('...nk,...mk->...nm',Q,K)/math.sqrt(d_k)
    if mask is not None:
        scores=scores.masked_fill(mask==False,float('-inf'))
        probs=softmax(scores,dim=-1)
        output=torch.einsum('...nm,...mv->...nv',probs,V)
        return output

