# =============================================================================
# core/embedding.py
# =============================================================================
import torch
import torch.nn as nn
import math
from config import MambaConfig

class MambaEmbedding(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (no positional encoding needed for Mamba)
        self.token_embedding = nn.Embedding(
            config.vocab_size, 
            config.d_model,
            dtype=config.dtype
        )
        
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        embeddings = self.token_embedding(input_ids)
        return embeddings 