# =============================================================================
# utils/selective_scan.py
# =============================================================================
import torch
import torch.nn.functional as F
from typing import Tuple

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """
    Selective scan function - core of Mamba's state space model
    
    Args:
        u: input sequence [batch, seq_len, d_inner]
        delta: time step [batch, seq_len, d_inner]
        A: state matrix [d_inner, d_state]
        B: input matrix [batch, seq_len, d_state]
        C: output matrix [batch, seq_len, d_state]
        D: skip connection [d_inner]
        z: gating [batch, seq_len, d_inner] (optional)
        delta_bias: bias for delta (optional)
        delta_softplus: whether to apply softplus to delta
    
    Returns:
        y: output [batch, seq_len, d_inner]
    """
    batch_size, seq_len, d_inner = u.shape
    d_state = A.shape[1]
    
    if delta_bias is not None:
        delta = delta + delta_bias[None, None, :]
    
    if delta_softplus:
        delta = F.softplus(delta)
    
    # Discretization
    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [batch, seq_len, d_inner, d_state]
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # [batch, seq_len, d_inner, d_state]
    
    # Initialize hidden state
    h = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
    
    outputs = []
    for i in range(seq_len):
        h = deltaA[:, i] * h + deltaB_u[:, i]  # State update
        y = torch.sum(h * C[:, i].unsqueeze(1), dim=-1)  # Output projection
        if D is not None:
            y = y + D * u[:, i]
        outputs.append(y)
    
    y = torch.stack(outputs, dim=1)  # [batch, seq_len, d_inner]
    
    if z is not None:
        y = y * F.silu(z)
    
    return y 