# =============================================================================
# training/optimizer.py
# =============================================================================
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Dict, List

class MambaOptimizer:
    """Optimizer setup for Mamba models"""
    
    def __init__(self, model, config):
        self.config = config
        self.model = model
        
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and layer norms
                if 'bias' in name or 'norm' in name or 'embedding' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                # Linear warmup
                return step / self.config.warmup_steps
            else:
                # Cosine decay
                progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def step(self):
        """Optimizer step with gradient clipping"""
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return self.scheduler.get_last_lr()[0]
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()
 