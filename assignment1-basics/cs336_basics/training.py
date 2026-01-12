import torch.nn.functional as F
from torch import Tensor
import os
import torch
from typing import BinaryIO, IO

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    Computes the cross entropy loss.
    inputs: (batch_size, vocab_size) - unnormalized logits
    targets: (batch_size,) - indices of correct classes
    """
    return F.cross_entropy(inputs, targets)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Saves a checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Loads a checkpoint and returns the iteration number.
    """
    checkpoint = torch.load(src, map_location="cpu") # Load to CPU first to be safe
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
