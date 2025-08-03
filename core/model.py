# =============================================================================
# core/model.py
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.config import MambaConfig
from core.embedding import MambaEmbedding
from core.mamba import MambaLayer, RMSNorm

class MambaModel(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embedding = MambaEmbedding(config)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaLayer(config) for _ in range(config.n_layers)
        ])
        
        # Final normalization
        self.norm_f = RMSNorm(config.d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights with embedding if specified
        if hasattr(config, 'tie_word_embeddings') and config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            input_ids: [batch, seq_len]
            targets: [batch, seq_len] (optional, for training)
        Returns:
            if targets is None: logits [batch, seq_len, vocab_size]
            else: (logits, loss)
        """
        # Get embeddings
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        
        # Apply Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        if targets is not None:
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            return logits, loss
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, 
                 temperature: float = 1.0, top_k: int = None):
        """Generate text autoregressively"""
        self.eval()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Get logits for last token
                logits = self.forward(input_ids)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self):
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 