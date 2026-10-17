import torch
import torch.nn as nn
import typing
import os
def load_checkpoint(
        src:typing.Union[str,os.PathLike,typing.BinaryIO,typing.IO[bytes]],
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer
)->int:
    #加载字典
    checkpoint=torch.load(src,map_location='cpu')
    #恢复模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    #恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']