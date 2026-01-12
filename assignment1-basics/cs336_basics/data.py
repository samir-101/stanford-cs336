import torch
import numpy as np

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of data from the dataset.
    dataset: 1D numpy array of token indices.
    Returns (x, y) where x is input and y is target (shifted by 1).
    """
    # Sample random starting indices
    # Must be at most len(dataset) - context_length - 1 to have x and y
    max_idx = len(dataset) - context_length - 1
    ix = np.random.randint(0, max_idx + 1, (batch_size,))
    
    # Collect batches
    x_list = []
    y_list = []
    
    for i in ix:
        # x: indices i to i+context_length
        # y: indices i+1 to i+context_length+1
        chunk_x = dataset[i : i + context_length]
        chunk_y = dataset[i + 1 : i + context_length + 1]
        x_list.append(chunk_x)
        y_list.append(chunk_y)
        
    x = torch.tensor(np.stack(x_list).astype(np.int64), device=device)
    y = torch.tensor(np.stack(y_list).astype(np.int64), device=device)
    
    return x, y
