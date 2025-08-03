# =============================================================================
# core/mamba.py
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.stateSpace import StateSpaceModel
from utils.conv_layer import Mamba1DConv

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight

class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Projections
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        
        # Convolution for local context
        self.conv1d = Mamba1DConv(config.d_inner, config.d_conv, config.conv_bias)
        
        # State space model
        self.ssm = StateSpaceModel(
            d_inner=config.d_inner,
            d_state=config.d_state,
            dt_rank=config.dt_rank,
            bias=config.bias
        )
        
        # Activation
        self.act = F.silu
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # [batch, seq_len, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)  # Each [batch, seq_len, d_inner]
        
        # Apply convolution
        x = self.act(self.conv1d(x))
        
        # Apply state space model
        y = self.ssm(x)
        
        # Apply gating with z
        y = y * self.act(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output

class MambaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mamba_block = MambaBlock(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        residual = x
        x = self.norm(x)
        x = self.mamba_block(x)
        return x + residual 