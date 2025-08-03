# =============================================================================
# core/stateSpace.py
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.selective_scan import selective_scan_fn

class StateSpaceModel(nn.Module):
    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = None, bias: bool = False):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(16, d_inner // 16)
        
        # State space parameters
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Projection layers
        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        # Initialize A with negative values for stability
        nn.init.uniform_(self.A_log, -4.0, -1.0)
        
        # Initialize dt_proj bias to encourage large dt values
        dt_init_std = self.dt_rank**-0.5
        with torch.no_grad():
            self.dt_proj.bias.uniform_(-dt_init_std, dt_init_std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_inner]
        Returns:
            y: [batch, seq_len, d_inner]
        """
        batch_size, seq_len, d_inner = x.shape
        
        # Project x to get delta, B, C
        x_dbl = self.x_proj(x)  # [batch, seq_len, dt_rank + 2*d_state]
        
        delta, B, C = torch.split(
            x_dbl, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1
        )
        
        # Project delta to d_inner
        delta = self.dt_proj(delta)  # [batch, seq_len, d_inner]
        
        # Get A matrix (ensure it's negative for stability)
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        
        # Apply selective scan
        y = selective_scan_fn(
            u=x,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=self.D,
            delta_softplus=True
        )
        
        return y 