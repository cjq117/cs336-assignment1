import torch
import torch.nn as nn
import cs336_basics.CausalSelfAttention as csa
import cs336_basics.RMSNorm as rms
import cs336_basics.SwiGLU as swiglu

class TransformerBlock(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,max_seq_len:int,theta:float,device=None,dtype=None):
        super().__init__()
        self.attn=csa.CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        #RMSNorm中有可学习的参数，所以初始化两个，分别应用于：attention和ffn
        self.ln1=rms.RMSNorm(d_model,device=device,dtype=dtype)
        self.ln2=rms.RMSNorm(d_model,device=device,dtype=dtype)
        self.ffn=swiglu.SwiGLU(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None)->torch.Tensor:
        #attention(pre_norm)
        x=x+self.attn(self.ln1(x),token_positions=token_positions)
        #ffn(pre_norm)
        x=x+self.ffn(self.ln2(x))
        return x
