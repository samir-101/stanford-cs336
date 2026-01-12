
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

def softmax(x: Tensor, dim: int) -> Tensor:
    """
    Computes the softmax of the input tensor along the specified dimension.
    """
    # Numerical stability is handled by PyTorch's softmax, but if we were to implement it:
    # x_max = x.max(dim=dim, keepdim=True).values
    # exp_x = torch.exp(x - x_max)
    # return exp_x / exp_x.sum(dim=dim, keepdim=True)
    return F.softmax(x, dim=dim)

def silu(x: Tensor) -> Tensor:
    """
    Computes the SiLU (Sigmoid Linear Unit) activation function.
    """
    return x * torch.sigmoid(x)

def rmsnorm(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-5
) -> Tensor:
    """
    Computes RMSNorm (Root Mean Square Layer Normalization).
    """
    # x shape: (..., d_model)
    # weight shape: (d_model,)
    
    # Calculate RMS
    pow_2 = x.pow(2)
    mean = pow_2.mean(dim=-1, keepdim=True)
    rms = torch.rsqrt(mean + eps)
    
    return x * rms * weight

def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None
) -> Tensor:
    """
    Applies a linear transformation y = xA^T + b.
    """
    return F.linear(x, weight, bias)

def embedding(
    input_ids: Tensor,
    weight: Tensor
) -> Tensor:
    """
    Retrieves embeddings for input_ids from the weight matrix.
    """
    return F.embedding(input_ids, weight)

def swiglu(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor
) -> Tensor:
    """
    Computes SwiGLU activation.
    w1: Gate projection
    w2: Down projection
    w3: Up projection
    """
    # x: (..., d_model)
    # w1, w3: (d_ff, d_model) -> Transposed in Linear layer usually, but here weights are passed directly
    # The adapter passes weights as (d_ff, d_model) for w1/w3 and (d_model, d_ff) for w2
    # So we use functional linear which expects (out, in)
    
    # Gate: x @ w1.T
    gate = F.linear(x, w1)
    # Value: x @ w3.T
    value = F.linear(x, w3)
    
    # Activation: silu(gate) * value
    hidden = F.silu(gate) * value
    
    # Output: hidden @ w2.T
    # w2 is (d_model, d_ff), so it maps back to d_model
    return F.linear(hidden, w2)

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Computes scaled dot product attention.
    q: (..., seq_len, d_k)
    k: (..., seq_len, d_k)
    v: (..., seq_len, d_v)
    mask: (..., seq_len, seq_len)
    """
    d_k = q.size(-1)
    # scores: (..., seq_len, seq_len)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=q.dtype))
    
    if mask is not None:
        # mask is boolean, True means mask out (ignore) -> fill with -inf
        # Or usually in PyTorch, mask True means keep.
        # Let's check the adapter doc or tests.
        # Test sends a boolean mask.
        # Usually attn_mask: 1/True to keep, 0/False to mask.
        # Or additive mask.
        # Let's assume standard PyTorch convention or check mask usage in tests using logic.
        # In `test_scaled_dot_product_attention`, mask is passed.
        # If mask is boolean:
        # If True -> participate. If False -> -inf.
        # Creating a large negative value for masking.
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """
    Computes multi-head self-attention.
    Weights are separate for q, k, v (not concatenated).
    """
    batch_size, seq_len, _ = in_features.shape
    d_head = d_model // num_heads
    
    # Project Q, K, V
    q = F.linear(in_features, q_proj_weight) 
    k = F.linear(in_features, k_proj_weight)
    v = F.linear(in_features, v_proj_weight)
    
    # Reshape for multi-head
    q = q.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    
    # Causal Mask
    # mask: (seq, seq). 1 to keep, 0 to mask?
    # scaled_dot expects: mask=0 -> -inf.
    # So we want 1s in lower triangle.
    mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device)).view(1, 1, seq_len, seq_len)
    
    # Scaled dot product attention
    output = scaled_dot_product_attention(q, k, v, mask=mask)
    
    # Reshape back
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    
    # Output projection
    return F.linear(output, o_proj_weight)

def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """
    Apply rotary embeddings to input tensor x.
    x: (batch, seq_len, num_heads, d_head) // or similar
    freqs_cis: (seq_len, d_head / 2) complex
    """
    # Shaping is tricky. Let's assume x is real, reshape to complex.
    # But usually we implement with sin/cos for better compatibility.
    
    # Let's use the explicit sin/cos implementation.
    # x: (..., seq_len, d_k)
    # We rotate pairs.
    d_k = x.shape[-1]
    x_pairs = x.view(*x.shape[:-1], -1, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]
    
    # freqs_cis (cos, sin) broadcasting.
    # We need to handle freqs.
    pass

def compute_freqs_cis(d_head: int, max_seq_len: int, theta: float = 10000.0) -> tuple[Tensor, Tensor]:
    # freq = 1 / (theta ** (2i / d))
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, d_head / 2)
    return torch.cos(freqs), torch.sin(freqs) 
    
def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    x: Tensor,
    token_positions: Tensor
) -> Tensor:
    """
    Applies RoPE to x.
    x: (..., seq_len, d_k)
    token_positions: (..., seq_len)
    """
    # 1. Compute frequencies
    # d_k must be even
    assert d_k % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float().to(x.device) / d_k))
    
    # 2. Get angles for positions
    # token_positions: (batch, seq) -> expand constraints
    # angles: (batch, seq, d_k/2)
    angles = torch.outer(token_positions.float().flatten(), freqs).view(*token_positions.shape, -1)
    
    # 3. cos/sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # 4. Repeat to match d_k (cos, cos, ..., sin, sin, ...) or (cos, sin, cos, sin...)
    # The standard implementation rotates pairs (x[i], x[i+1]).
    # So we need cos/sin for each pair.
    # angles corresponds to indices 0, 2, 4... in the embedding.
    # so we repeat interleave or repeat?
    # if x is shaped (..., d_k/2, 2), then pairs are the last dim.
    # cos/sin are (..., d_k/2).
    
    # Reshape x to (..., d_k/2, 2)
    x_shaped = x.view(*x.shape[:-1], -1, 2)
    x_real = x_shaped[..., 0]
    x_imag = x_shaped[..., 1]
    
    # cos, sin are (..., d_k/2). Broadcast to (..., d_k/2)
    # Apply rotation:
    # x' = x cos - y sin
    # y' = x sin + y cos
    
    # Expand shapes if necessary? cos is (batch, seq, d_k/2) matching x_real/imag
    
    x_out_real = x_real * cos - x_imag * sin
    x_out_imag = x_real * sin + x_imag * cos
    
    x_out = torch.stack([x_out_real, x_out_imag], dim=-1).flatten(start_dim=-2)
    return x_out

def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Optional[Tensor] = None,
) -> Tensor:
    batch_size, seq_len, _ = in_features.shape
    d_head = d_model // num_heads
    
    # Projections
    q = F.linear(in_features, q_proj_weight)
    k = F.linear(in_features, k_proj_weight)
    v = F.linear(in_features, v_proj_weight)
    
    # Reshape
    # (batch, seq, num_heads * d_head) -> ...
    q = q.view(batch_size, seq_len, num_heads, d_head)
    k = k.view(batch_size, seq_len, num_heads, d_head)
    v = v.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    
    # Apply RoPE to Q and K
    # Transpose to (batch, num_heads, seq, d_head) for attn?
    # Usually RoPE is applied on the sequence dimension.
    # My rope function expects (..., seq, d_k).
    # q is currently (batch, seq, num_heads, d_head). 
    # We can treat batch*num_heads as batch? 
    # Or just permute: (batch, num_heads, seq, d_head)
    
    q = q.transpose(1, 2) # (batch, num_heads, seq, d_head)
    k = k.transpose(1, 2)
    
    if token_positions is not None:
        # token_positions: (batch, seq).
        # We need to broadcast to (batch, num_heads, seq) or similar.
        # My rope implementation expects x and positions to broadcast.
        # x: (batch, num_heads, seq, d_head)
        # positions: (batch, seq) -> (batch, 1, seq) -> broadcast?
        
        # RoPE function check:
        # angles = outer(pos, freq) -> (batch, seq, d_head/2)
        # cos/sin -> (batch, seq, d_head/2)
        # x -> (batch, num_heads, seq, d_head/2, 2) where d_head/2 is dim -2.
        # cos needs to unsqueeze for num_heads: (batch, 1, seq, d_head/2)
        
        # Let's write a targeted apply_rope helper here or update rope()
        
        pos_expanded = token_positions.unsqueeze(1) # (batch, 1, seq)
        # freq calculation
        freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float().to(q.device) / d_head)) # (d_head/2)
        
        # pos indices (batch, 1, seq) * freqs (d_head/2)
        # outer product manually via broadcast
        # (batch, 1, seq, 1) * (d_head/2) -> (batch, 1, seq, d_head/2)
        angles = pos_expanded.unsqueeze(-1) * freqs
        
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        def rotate(x_in):
            # x_in: (batch, num_heads, seq, d_head)
            x_pairs = x_in.view(*x_in.shape[:-1], -1, 2)
            r = x_pairs[..., 0]
            i = x_pairs[..., 1]
            r_out = r * cos - i * sin
            i_out = r * sin + i * cos
            return torch.stack([r_out, i_out], dim=-1).flatten(start_dim=-2)
            
        q = rotate(q)
        k = rotate(k)
        
    # Standard MHA
    # Causal mask needed here too
    mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device)).view(1, 1, seq_len, seq_len)
    output = scaled_dot_product_attention(q, k, v, mask=mask)
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    return F.linear(output, o_proj_weight)

def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    """
    Implements a Transformer Block with RoPE, pre-norm, and SwiGLU.
    """
    # Pre-Norm 1 + Attn
    x = in_features
    norm1 = rmsnorm(x, weights["ln1.weight"], eps=1e-5)
    
    # Attn
    # Need to extract q,k,v,o weights from dict
    # Dictionary keys are prefix-less relative to block
    attn_out = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=norm1,
        token_positions=torch.arange(in_features.size(1), device=in_features.device).expand(in_features.size(0), -1)
    )
    
    x = x + attn_out
    
    # Pre-Norm 2 + FFN
    norm2 = rmsnorm(x, weights["ln2.weight"], eps=1e-5)
    
    ffn_out = swiglu(
        norm2,
        w1=weights["ffn.w1.weight"],
        w2=weights["ffn.w2.weight"],
        w3=weights["ffn.w3.weight"]
    )
    
    x = x + ffn_out
    return x

def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    """
    Implements a Transformer Language Model.
    """
    # Embeddings
    x = F.embedding(in_indices, weights["token_embeddings.weight"])
    
    # Layers
    for i in range(num_layers):
        # Extract weights for this layer using prefix `layers.{i}.`
        layer_prefix = f"layers.{i}."
        block_weights = {
            k.replace(layer_prefix, ""): v 
            for k, v in weights.items() 
            if k.startswith(layer_prefix)
        }
        
        x = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=block_weights,
            in_features=x
        )
        
    # Final Norm
    x = rmsnorm(x, weights["ln_final.weight"], eps=1e-5)
    
    # LM Head
    logits = F.linear(x, weights["lm_head.weight"])
    return logits

