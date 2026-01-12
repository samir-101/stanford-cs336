
import torch
from typing import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clips the gradients of the parameters to a maximum L2 norm.
    Gradients are modified in-place.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
    )
    
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

def get_adamw_cls():
    """
    Returns the AdamW optimizer class.
    """
    return torch.optim.AdamW

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Computes learning rate at iteration `it` using cosine schedule with warmup.
    """
    # 1. Warmup
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    # 2. Cosine Decay
    if it < cosine_cycle_iters:
        # Progress from 0 to 1 within the cosine cycle (after warmup)
        # Wait, usually cosine cycle starts from 0? or from warmup_end?
        # Standard: from warmup_end to cosine_cycle_iters?
        # The prompt says: "cosine_cycle_iters (int): T_c, the number of cosine annealing iterations."
        # Usually total iterations = T_c.
        
        # Fraction of progress
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        import math
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * progress)
        )
        
    # 3. Post-cycle (min LR)
    return min_learning_rate
