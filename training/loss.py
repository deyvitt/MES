# =============================================================================
# training/loss.py
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class MambaLoss(nn.Module):
    """Loss functions for Mamba training"""
    
    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Primary loss
        self.lm_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Auxiliary losses
        self.diversity_weight = 0.01
        self.specialist_balance_weight = 0.001
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                specialist_weights: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss
        
        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
            specialist_weights: Dict of specialist activation weights
            
        Returns:
            Dict with loss components
        """
        losses = {}
        
        # Primary language modeling loss
        lm_loss = self.lm_loss(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        losses['lm_loss'] = lm_loss
        
        # Diversity loss to encourage specialist specialization
        if specialist_weights is not None:
            diversity_loss = self._compute_diversity_loss(specialist_weights)
            losses['diversity_loss'] = diversity_loss
            
            # Balance loss to prevent specialist dominance
            balance_loss = self._compute_balance_loss(specialist_weights)
            losses['balance_loss'] = balance_loss
        else:
            losses['diversity_loss'] = torch.tensor(0.0, device=logits.device)
            losses['balance_loss'] = torch.tensor(0.0, device=logits.device)
        
        # Total loss
        total_loss = (
            lm_loss + 
            self.diversity_weight * losses['diversity_loss'] +
            self.specialist_balance_weight * losses['balance_loss']
        )
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_diversity_loss(self, specialist_weights: Dict) -> torch.Tensor:
        """Encourage specialists to be diverse"""
        if len(specialist_weights) < 2:
            return torch.tensor(0.0)
        
        # Convert weights to tensor
        weights = torch.stack(list(specialist_weights.values()))
        
        # Compute pairwise similarities
        normalized_weights = F.normalize(weights, dim=-1)
        similarity_matrix = torch.mm(normalized_weights, normalized_weights.t())
        
        # Penalize high similarities (encourage diversity)
        diversity_loss = similarity_matrix.triu(diagonal=1).mean()
        
        return diversity_loss
    
    def _compute_balance_loss(self, specialist_weights: Dict) -> torch.Tensor:
        """Encourage balanced specialist usage"""
        if not specialist_weights:
            return torch.tensor(0.0)
        
        # Get activation frequencies
        activations = torch.stack(list(specialist_weights.values()))
        
        # Compute variance in activations (lower is more balanced)
        balance_loss = activations.var()
        
        return balance_loss 