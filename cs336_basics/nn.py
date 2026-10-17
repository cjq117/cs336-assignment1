import torch
import torch.nn as nn
from cs336_basics.Embedding import Embedding
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Linear import Linear

class TransformerLM(nn.Module):
    def __init__(self,vocab_size:int,max_seq_len:int,d_model:int,num_layers:int,
                 num_heads:int,d_ff:int,rope_theta:float,device=None,dtype=None,
            #新增实验参数
            use_rms_norm:bool=True,
            norm_mode:str="pre",
            ffn_type:str="swiglu"):
        super().__init__()
        self.max_seq_len=max_seq_len
        self.token_embeddings=Embedding(vocab_size,d_model,device=device,dtype=dtype)
        self.layers=nn.ModuleList([
            TransformerBlock(
                d_model,num_heads,d_ff,max_seq_len,rope_theta,
                device=device,
                dtype=dtype,
                use_rms_norm=use_rms_norm,
                norm_mode=norm_mode,
                ffn_type=ffn_type
            )
            for _ in range(num_layers)
        ])
        #如果全局禁用了Norm,这里的Final,Norm也要变成Identity
        if use_rms_norm:
            self.ln_final=RMSNorm(d_model,device=device,dtype=dtype)
        else:
            self.ln_final=nn.Identity()
        #经过Linear层映射回词表大小
        self.lm_head=Linear(d_model,vocab_size,device=device,dtype=dtype)
    
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        b,s=token_ids.shape
        token_positions=torch.arange(s,device=token_ids.device).unsqueeze(0).expand(b,s)
        x=self.token_embeddings(token_ids)
        for layer in self.layers:
            x=layer(x,token_positions=token_positions)
        x=self.ln_final(x)
        return self.lm_head(x)


    
     

