import torch
from collections.abc import Iterable
#w1,b1,w2,b2......取
def clip_gradient_norm(parameters:Iterable[torch.nn.Parameter],max_norm:float):
     #过滤掉没有梯度的参数
    params_with_grad=[p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    total_norm=0.0
    for p in params_with_grad:
        #.detach()让后续裁剪操作脱离计算图。
        #算出当前参数张量的梯度的L2范数
        param_norm=torch.norm(p.grad.detach(),p=2)
        total_norm+=param_norm.item()**2
    total_norm=total_norm**0.5
    eps=1e-6
    if total_norm>max_norm:
        clip_coef=max_norm/(total_norm+eps)
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)