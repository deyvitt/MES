#!/usr/bin/env python3
"""
Mamba Encoder Swarm - Integration with Existing Mamba Implementation
Uses your existing Mamba components as building blocks for the swarm architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# Import your existing Mamba components
from core.config import MambaConfig
from core.model import MambaModel
from core.mamba import MambaLayer, RMSNorm
from core.embedding import MambaEmbedding

class SwarmRouter(nn.Module):
    """
    Routes input tokens to different encoder instances
    This is the NEW component that enables the swarm architecture
    """
    
    def __init__(self, d_model: int, num_encoders: int, routing_strategy: str = "learned"):
        super().__init__()
        self.d_model = d_model
        self.num_encoders = num_encoders
        self.routing_strategy = routing_strategy
        
        if routing_strategy == "learned":
            # Neural router that learns optimal token distribution
            self.router_network = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, num_encoders),
                nn.Softmax(dim=-1)
            )
        
        # Load balancing coefficient
        self.load_balance_coef = 0.01
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Route tokens to encoder instances
        
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            encoder_inputs: List of inputs for each encoder
            routing_weights: Weights for aggregation [batch, seq_len, num_encoders]
            load_balance_loss: Loss term for training
        """
        batch_size, seq_len, d_model = x.shape
        
        if self.routing_strategy == "learned":
            # Learn routing patterns
            routing_logits = self.router_network(x)  # [batch, seq_len, num_encoders]
            routing_weights = F.gumbel_softmax(routing_logits, tau=1.0, hard=False)
            
            # Load balancing loss to encourage equal usage
            avg_routing = routing_weights.mean(dim=[0, 1])
            load_balance_loss = self.load_balance_coef * torch.var(avg_routing)
            
        else:  # Round-robin for simplicity
            seq_indices = torch.arange(seq_len, device=x.device)
            encoder_ids = seq_indices % self.num_encoders
            routing_weights = F.one_hot(encoder_ids, self.num_encoders).float()
            routing_weights = routing_weights.unsqueeze(0).expand(batch_size, -1, -1)
            load_balance_loss = torch.tensor(0.0, device=x.device)
        
        # Create weighted inputs for each encoder
        encoder_inputs = []
        for i in range(self.num_encoders):
            weight = routing_weights[:, :, i:i+1]  # [batch, seq_len, 1]
            encoder_input = x * weight
            encoder_inputs.append(encoder_input)
        
        return encoder_inputs, routing_weights, load_balance_loss

class SwarmAggregator(nn.Module):
    """
    Aggregates outputs from all encoder instances
    This is the NEW component that combines swarm outputs
    """
    
    def __init__(self, d_model: int, num_encoders: int):
        super().__init__()
        self.d_model = d_model
        self.num_encoders = num_encoders
        
        # Attention-based aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True
        )
        
        # Output processing
        self.norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, encoder_outputs: List[torch.Tensor], routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Aggregate encoder outputs using learned attention
        
        Args:
            encoder_outputs: List of [batch, seq_len, d_model] tensors
            routing_weights: [batch, seq_len, num_encoders]
            
        Returns:
            aggregated: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = encoder_outputs[0].shape
        
        # Stack and weight encoder outputs
        stacked = torch.stack(encoder_outputs, dim=2)  # [batch, seq_len, num_encoders, d_model]
        routing_expanded = routing_weights.unsqueeze(-1)  # [batch, seq_len, num_encoders, 1]
        weighted = stacked * routing_expanded
        
        # Initial aggregation
        initial = weighted.sum(dim=2)  # [batch, seq_len, d_model]
        
        # Attention-based refinement
        encoder_sequence = stacked.view(batch_size, seq_len * self.num_encoders, d_model)
        refined, _ = self.attention(initial, encoder_sequence, encoder_sequence)
        
        # Final processing
        output = self.output_proj(refined)
        output = self.norm(output + initial)  # Residual connection
        
        return output

class MambaEncoderSwarmModel(nn.Module):
    """
    Complete Swarm Model using your existing Mamba components
    
    Architecture:
    1. Use your MambaEmbedding for input processing
    2. NEW: Router distributes tokens to encoder swarm
    3. Use your MambaLayer instances as shared encoders  
    4. NEW: Aggregator combines encoder outputs
    5. Use your MambaLayer instances for decoder
    6. Use your existing LM head for output
    """
    
    def __init__(self, config: MambaConfig, num_encoders: int = 8, routing_strategy: str = "learned"):
        super().__init__()
        self.config = config
        self.num_encoders = num_encoders
        
        # Use your existing embedding
        self.embedding = MambaEmbedding(config)
        
        # NEW: Swarm components
        self.router = SwarmRouter(config.d_model, num_encoders, routing_strategy)
        
        # Shared encoder (using your MambaLayer)
        # All encoder instances will use this same layer (weight sharing!)
        self.shared_encoder_layer = MambaLayer(config)
        
        # NEW: Aggregator
        self.aggregator = SwarmAggregator(config.d_model, num_encoders)
        
        # Decoder layers (using your MambaLayer)
        self.decoder_layers = nn.ModuleList([
            MambaLayer(config) for _ in range(config.n_layers)
        ])
        
        # Use your existing components
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
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
        Forward pass through swarm architecture
        
        Args:
            input_ids: [batch, seq_len]
            targets: [batch, seq_len] (optional, for training)
            
        Returns:
            if targets is None: logits [batch, seq_len, vocab_size]
            else: (logits, loss, load_balance_loss)
        """
        # 1. Embedding (using your existing component)
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        
        # 2. Route to encoder swarm
        encoder_inputs, routing_weights, load_balance_loss = self.router(x)
        
        # 3. Process through shared encoder instances
        encoder_outputs = []
        for encoder_input in encoder_inputs:
            # Each instance uses the SAME shared_encoder_layer (weight sharing!)
            encoder_output = self.shared_encoder_layer(encoder_input)
            encoder_outputs.append(encoder_output)
        
        # 4. Aggregate encoder outputs
        x = self.aggregator(encoder_outputs, routing_weights)
        
        # 5. Process through decoder (using your existing layers)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        
        # 6. Final processing (using your existing components)
        x = self.norm_f(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        if targets is not None:
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            return logits, loss, load_balance_loss
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, 
                 temperature: float = 1.0, top_k: int = None):
        """Generate using swarm architecture"""
        self.eval()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(input_ids)
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self):
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_swarm_from_existing_config(config: MambaConfig, num_encoders: int = 8) -> MambaEncoderSwarmModel:
    """
    Create swarm model using your existing configuration
    """
    swarm_model = MambaEncoderSwarmModel(config, num_encoders, routing_strategy="learned")
    
    num_params = swarm_model.get_num_params()
    print(f"🚀 Swarm model created with {num_params:,} parameters ({num_params/1e6:.1f}M)")
    print(f"📊 Using {num_encoders} encoder instances with shared weights")
    
    return swarm_model

def compare_architectures(config: MambaConfig):
    """
    Compare standard Mamba vs Swarm architecture
    """
    print("🔍 Architecture Comparison")
    print("=" * 50)
    
    # Standard model (your existing)
    standard_model = MambaModel(config)
    standard_params = standard_model.get_num_params()
    
    # Swarm model (new architecture)
    swarm_model = create_swarm_from_existing_config(config, num_encoders=8)
    swarm_params = swarm_model.get_num_params()
    
    print(f"📈 Standard Mamba: {standard_params:,} parameters ({standard_params/1e6:.1f}M)")
    print(f"🔥 Swarm Mamba:    {swarm_params:,} parameters ({swarm_params/1e6:.1f}M)")
    print(f"💡 Parameter overhead: {((swarm_params - standard_params) / standard_params * 100):.1f}%")
    
    return standard_model, swarm_model

if __name__ == "__main__":
    # Test with your existing config
    from core.config import MambaConfig
    
    # Create a test config
    config = MambaConfig(
        vocab_size=50257,
        d_model=512,
        n_layers=8,
        d_state=16,
        d_conv=4,
        bias=False
    )
    
    print("🧪 Testing Swarm Integration")
    print("=" * 40)
    
    # Compare architectures
    standard_model, swarm_model = compare_architectures(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test standard model
    with torch.no_grad():
        standard_logits = standard_model(input_ids)
        print(f"✅ Standard model output: {standard_logits.shape}")
    
    # Test swarm model
    with torch.no_grad():
        swarm_logits = swarm_model(input_ids)
        print(f"✅ Swarm model output: {swarm_logits.shape}")
    
    print(f"\n🎉 Both architectures working! Ready to train the swarm.") 