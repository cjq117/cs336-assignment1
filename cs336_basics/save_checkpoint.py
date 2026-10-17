import torch
import torch.nn as nn
import typing
import os


def save_checkpoint(
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        iteration:int,
        out:typing.Union[str,os.PathLike,typing.BinaryIO,typing.IO[bytes]]
):
    checkpoint={
        #获取模型当前的状态字典，包含权重等数据
        'model_state_dict':model.state_dict(),
        #获取优化器当前的状态字典，放在 'optimizer_state_dict' 这个键下面
        'optimizer_state_dict':optimizer.state_dict(),
        'iteration':iteration
    }
    torch.save(checkpoint,out)
    