# =============================================================================
# routing/aggregator.py
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from core.config import MambaConfig

class AttentionAggregator(nn.Module):
    """Attention-based aggregator for combining specialist outputs"""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_specialists = config.num_specialists
        
        # Attention mechanism for combining specialist outputs
        self.specialist_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Project specialist confidence scores
        self.confidence_proj = nn.Linear(1, self.d_model)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Final language modeling head
        self.lm_head = nn.Linear(self.d_model, config.vocab_size, bias=False)
        
    def forward(self, specialist_outputs: Dict[int, List[Dict]]) -> torch.Tensor:
        """
        Aggregate specialist outputs into final representation
        
        Args:
            specialist_outputs: Dict mapping chunk_id to list of specialist results
            
        Returns:
            aggregated_logits: [batch, seq_len, vocab_size]
        """
        batch_outputs = []
        
        for chunk_id in sorted(specialist_outputs.keys()):
            chunk_results = specialist_outputs[chunk_id]
            
            if not chunk_results:
                continue
            
            # Stack specialist encodings
            encodings = []
            confidences = []
            
            for result in chunk_results:
                if result is not None:
                    encodings.append(result['encoding'])
                    confidences.append(result['confidence'])
            
            if not encodings:
                continue
            
            # Stack tensors
            specialist_encodings = torch.stack(encodings)  # [num_specialists, d_model]
            confidence_scores = torch.tensor(confidences, device=encodings[0].device)
            
            # Project confidence scores
            confidence_embeddings = self.confidence_proj(
                confidence_scores.unsqueeze(-1)
            )  # [num_specialists, d_model]
            
            # Add confidence information to encodings
            enhanced_encodings = specialist_encodings + confidence_embeddings
            
            # Apply attention to combine specialist outputs
            # Use self-attention to let specialists communicate
            aggregated, _ = self.specialist_attention(
                enhanced_encodings.unsqueeze(0),  # [1, num_specialists, d_model]
                enhanced_encodings.unsqueeze(0),
                enhanced_encodings.unsqueeze(0)
            )
            
            # Pool the attended representations
            chunk_representation = aggregated.mean(dim=1)  # [1, d_model]
            
            # Apply output layers
            chunk_output = self.output_layers(chunk_representation)
            batch_outputs.append(chunk_output)
        
        if not batch_outputs:
            # Return dummy output if no valid results
            return torch.zeros(1, 1, self.config.vocab_size)
        
        # Concatenate chunk outputs
        final_representation = torch.cat(batch_outputs, dim=0)  # [num_chunks, d_model]
        
        # Generate logits
        logits = self.lm_head(final_representation)  # [num_chunks, vocab_size]
        
        return logits.unsqueeze(0)  # [1, num_chunks, vocab_size]
    
    def generate_response(self, specialist_outputs: Dict[int, List[Dict]], 
                         max_tokens: int = 100) -> str:
        """Generate text response from specialist outputs"""
        # Get aggregated logits
        logits = self.forward(specialist_outputs)
        
        # Simple greedy decoding (can be improved with better generation)
        generated_ids = []
        current_logits = logits[0, -1, :]  # Use last chunk's logits
        
        for _ in range(max_tokens):
            # Get next token
            next_token = torch.argmax(current_logits, dim=-1)
            generated_ids.append(next_token.item())
            
            # Break on EOS token (assuming token 0 is EOS)
            if next_token.item() == 0:
                break
        
        # Convert to text (placeholder - should use proper tokenizer)
        # This is simplified - integrate with actual tokenizer for real text
        response = f"Generated response with {len(generated_ids)} tokens"
        
        return response 