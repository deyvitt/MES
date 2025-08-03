# =============================================================================
# core/config.py
# =============================================================================
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MambaConfig:
    # Model architecture
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 12
    d_inner: int = 2048
    d_state: int = 16
    d_conv: int = 4
    dt_rank: Optional[int] = None
    bias: bool = False
    conv_bias: bool = True
    
    # Training
    max_seq_len: int = 2048
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Swarm specific
    num_specialists: int = 100
    specialist_domains: List[str] = None
    shared_embedding: bool = True
    hierarchical_sharing: bool = True
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    
    def __post_init__(self):
        if self.dt_rank is None:
            self.dt_rank = max(16, self.d_model // 16)
        if self.specialist_domains is None:
            self.specialist_domains = [f"domain_{i}" for i in range(self.num_specialists)]
 