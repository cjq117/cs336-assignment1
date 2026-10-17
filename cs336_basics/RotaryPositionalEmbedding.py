import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    #d_k每个head的维度（必须是偶数）
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.d_k=d_k
        #powers是theta的指数
        powers=torch.arange(0,d_k,2,device=device).float()/d_k
        freqs=1.0/(theta**powers)
        #t是i的序列[0,1,...,max_seq_len-1]
        t=torch.arange(max_seq_len,device=device).float()
        #外积，生成i和freqs（频率）的矩阵
        #这个矩阵存放的是各位置、各组分量需要旋转的角度
        freqs_matrix=torch.outer(t,freqs)
        #persistent=False表示这些缓冲区不会在state_dict中出现（可以随时重新生成）
        #得到的是那个稀疏矩阵的简化版本（更高效更快）
        #这些缓存由 theta、位置和维度按照公式确定

        '''
        freqs_matrix = [      
          [0, 0.0],
          [1, 0.1],
          [2, 0.2]
        ]
        
        cos_cache ≈ [
          [ 1.0000, 1.0000],
          [ 0.5403, 0.9950],
          [-0.4161, 0.9801]
        ]

        sin_cache ≈ [
         [0.0000, 0.0000],
         [0.8415, 0.0998],
         [0.9093, 0.1987]
        ] 
        '''
        self.register_buffer("cos_cache",freqs_matrix.cos(),persistent=False)
        self.register_buffer("sin_cache",freqs_matrix.sin(),persistent=False)
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        cos=self.cos_cache[token_positions]
        sin=self.sin_cache[token_positions]
        #维度对齐
        if x.ndim>cos.ndim and cos.ndim>=3:
            cos=cos.unsqueeze(1)
            sin=sin.unsqueeze(1)
        #确保类型一致
        cos=cos.to(x.dtype)
        sin=sin.to(x.dtype)
        #拆分并旋转
        x_even=x[...,0::2]
        x_odd=x[...,1::2]
        output=torch.empty_like(x)
        output[...,0::2]=x_even*cos-x_odd*sin
        output[...,1::2]=x_even*sin+x_odd*cos
        return output
                




