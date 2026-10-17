import torch
import torch.nn as nn
from einops import rearrange
from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.Linear import Linear
from cs336_basics.RotaryPositionalEmbedding import RotaryPositionalEmbedding

class CausalSelfAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,max_seq_len=None,theta=None,device=None,dtype=None):
        super().__init__()
        assert d_model%num_heads==0
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        #创建四个线性层
        #q_proj：把输入映射成Query,参与决定当前token如何寻找相关信息
        #k_proj：把输入映射成key，用来与Query计算匹配分数
        #v_proj:把输入映射成value，提供最后被加权汇总的信息
        #output_proj:把多个头合并后的结果再做一次线性变换，融合各个头的信息
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)


        #RoPE初始化
        if theta is not None and max_seq_len is not None:
            self.rope=RotaryPositionalEmbedding(theta,self.d_k,max_seq_len,device=device)
        else:
            self.rope=None


    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None)->torch.Tensor:
        b,s,d=x.shape
        q=rearrange(self.q_proj(x),'... s(h d)->... h s d',h=self.num_heads)
        k=rearrange(self.k_proj(x),'... s(h d)->... h s d',h=self.num_heads)
        v=rearrange(self.v_proj(x),'... s(h d)->... h s d',h=self.num_heads)

        #应用旋转位置编码
        if self.rope is not None:
           if token_positions is None:
               token_positions=torch.arange(s,device=x.device).expand(b,s)
           #只对q,k添加位置信息，因为Q*K（转置）时，不同Tokens之间交互信息才添加位置信息    
           q=self.rope(q,token_positions)
           k=self.rope(k,token_positions)

        #生成因果掩码（下三角矩阵，下三角是True,上三角是False）
        mask=torch.tril(torch.ones(s,s,device=x.device,dtype=torch.bool))
        #这里的q,k,v指的是多头拆分之后的q,k,v
        attn_out=scaled_dot_product_attention(q,k,v,mask=mask)
        #合并多头
        attn_out=rearrange(attn_out,'... h s d->... s (h d)')
        return self.output_proj(attn_out)
