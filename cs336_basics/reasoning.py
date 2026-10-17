import torch
from cs336_basics.softmax import softmax

def _top_p_filter(logits:torch.Tensor,p:float)->torch.Tensor:
   sorted_logits,sorted_indices=torch.sort(logits,descending=True,dim=-1)
   cumulative_probs=torch.cumsum(softmax(sorted_logits,dim=-1),dim=-1)
   sorted_indices_to_remove=cumulative_probs>p
   sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
   sorted_indices_to_remove[...,0]=False
   indices_to_remove=sorted_indices_to_remove.scatter(1,sorted_indices,sorted_indices_to_remove)
   logits=logits.masked_fill(indices_to_remove,float('-inf'))
   return logits
@torch.no_grad()
def generate(
    self,
    prompt_ids:torch.Tensor,
    max_new_tokens:int,
    eos_token_id:int=None,
    temperature:float=1.0,
    top_p:float=1.0
)->torch.Tensor:
    self.eval()
    generated=prompt_ids.clone()
    for _ in range(max_new_tokens):
      idx_cond=generated[:,-self.max_seq_len:]
      logits=self.forward(idx_cond)
      logits=logits[:,-1,:]
      if(temperature!=1.0):
         logits=logits/(temperature+1e-8)
      if top_p<1.0:
         logits=_top_p_filter(logits,top_p)
      probs=softmax(logits,dim=-1)
      next_token=torch.multinomial(probs,num_samples=1)
      generated=torch.cat((generated,next_token),dim=-1)
      if eos_token_id is not None and (next_token==eos_token_id).all():
         break
          
    return generated



