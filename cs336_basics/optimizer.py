import torch
import math
from torch.optim import Optimizer

#根据loss.backward() 算好的梯度，用 AdamW 更新模型参数
class AdamW(Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        if lr<0.0:
            raise ValueError(f"Invalid learning rate:{lr}")
        if not 0.0<=betas[0]<1.0:
            raise ValueError(f"Invalid beta parameter at index 0:{betas[0]}")
        if not 0.0<=betas[1]<1.0:
            raise ValueError(f"Invalid beta parameter at index 1:{betas[1]}")
        if eps<0.0:
            raise ValueError(f"Invalid epsilon value:{eps}")

        defaults=dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super().__init__(params,defaults)
    #@torch.no_grad()：更新参数时，不把这些更新操作记录进用于反向传播的计算图。
    @torch.no_grad()
    def step(self):
            loss=None
            for group in self.param_groups:
                beta1,beta2=group['betas']
                eps=group['eps']
                lr=group['lr']
                wd=group['weight_decay']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad=p.grad
                    #为每个参数保存历史记录
                    state=self.state[p]
                    #如果记录为空，说明这个参数还没有进行过更新，需要初始化
                    if len(state)==0:
                        #记录这个参数已经更新了多少次，初始为 0。
                        state['step']=0
                        #创建与 p 形状、设备、数据类型一致的全零张量，用来保存一阶动量m
                        #memory_format=torch.preserve_format 用于尽量保留原参数的内存布局。
                        state['exp_avg']=torch.zeros_like(p,memory_format=torch.preserve_format)
                        state['exp_avg_sq']=torch.zeros_like(p,memory_format=torch.preserve_format)
                    exp_avg,exp_avg_sq=state['exp_avg'],state['exp_avg_sq']
                    state['step']+=1
                    t=state['step']
                    #m=beta1*m+(1-beta1)*g
                    exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)
                    #函数末尾的 _ 表示直接修改原张量
                    #v=beta2*v+(1-beta2)*g^2
                    exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                    bias_correction1=1-beta1**t
                    bias_correction2=1-beta2**t
                    step_size=lr*(math.sqrt(bias_correction2)/bias_correction1)
                    denom=exp_avg_sq.sqrt().add_(eps)
                    if(wd)!=0:
                        #如果启用了权重衰减，就执行：p=p-p*lr*wd
                        p.add_(p,alpha=-lr*wd)
                    #addcdiv_ 会直接更新参数；addcdiv 返回新张量，如果没有接收返回值，这一步更新就不会保存到 p
                    p.addcdiv_(exp_avg,denom,value=-step_size)
            return loss





