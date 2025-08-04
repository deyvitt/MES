# modeling_mamba_swarm.py - HuggingFace integration for Mamba Swarm

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class MambaSwarmConfig(PretrainedConfig):
    """Configuration class for MambaSwarm model"""
    model_type = "mamba_swarm"
    
    def __init__(
        self,
        num_encoders=100,
        max_mamba_encoders=100,
        d_model=768,
        vocab_size=50257,
        max_sequence_length=2048,
        encoder_config=None,
        router_config=None,
        aggregator_config=None,
        **kwargs
    ):
        self.num_encoders = num_encoders
        self.max_mamba_encoders = max_mamba_encoders
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.encoder_config = encoder_config or {}
        self.router_config = router_config or {}
        self.aggregator_config = aggregator_config or {}
        super().__init__(**kwargs)

class MambaSwarmForCausalLM(PreTrainedModel):
    """HuggingFace compatible Mamba Swarm model"""
    config_class = MambaSwarmConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Initialize core components
        try:
            # Try to use the unified swarm engine
            from system.mambaSwarm import UnifiedMambaSwarm
            self.swarm_engine = UnifiedMambaSwarm(
                config=config,
                use_pretrained=False  # Use native implementation
            )
            self.num_active_encoders = getattr(self.swarm_engine, 'num_encoders', config.num_encoders)
            logger.info("Initialized with UnifiedMambaSwarm")
            
        except ImportError:
            try:
                # Fallback to native swarm integration
                from core.mamba_swarm_integration import MambaEncoderSwarmModel
                from core.config import MambaConfig
                
                # Convert config to MambaConfig
                mamba_config = MambaConfig(
                    d_model=config.d_model,
                    vocab_size=config.vocab_size,
                    n_layers=8,  # Default
                    d_state=16,  # Default
                    d_conv=4,    # Default
                    bias=False   # Default
                )
                
                self.swarm_engine = MambaEncoderSwarmModel(
                    mamba_config, 
                    num_encoders=config.num_encoders
                )
                self.num_active_encoders = config.num_encoders
                logger.info("Initialized with MambaEncoderSwarmModel")
                
            except ImportError as e:
                logger.error(f"Could not import swarm components: {e}")
                # Create a minimal mock implementation
                self.swarm_engine = self._create_mock_engine(config)
                self.num_active_encoders = config.num_encoders
                logger.warning("Using mock swarm engine")
    
    def _create_mock_engine(self, config):
        """Create a mock engine for testing purposes"""
        class MockSwarmEngine:
            def __init__(self, config):
                self.config = config
                self.embedding = nn.Embedding(config.vocab_size, config.d_model)
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
                self.num_active_encoders = config.num_encoders
            
            def forward(self, input_ids, **kwargs):
                # Simple passthrough for testing
                embeddings = self.embedding(input_ids)
                logits = self.lm_head(embeddings)
                return type('MockOutput', (), {'logits': logits, 'past_key_values': None})()
            
            def generate(self, input_ids, max_length=100, **kwargs):
                # Simple generation for testing
                batch_size, seq_len = input_ids.shape
                new_tokens = torch.randint(0, self.config.vocab_size, (batch_size, max_length - seq_len))
                return torch.cat([input_ids, new_tokens], dim=1)
            
            def set_active_encoders(self, num):
                self.num_active_encoders = min(num, self.config.max_mamba_encoders)
        
        return MockSwarmEngine(config)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """Forward pass through the swarm model"""
        
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        # Get outputs from swarm engine
        if hasattr(self.swarm_engine, 'forward'):
            outputs = self.swarm_engine.forward(input_ids, **kwargs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        else:
            # Fallback for engines without forward method
            try:
                logits = self.swarm_engine(input_ids)
            except Exception as e:
                logger.error(f"Forward pass failed: {e}")
                # Emergency fallback
                batch_size, seq_len = input_ids.shape
                logits = torch.randn(batch_size, seq_len, self.config.vocab_size)
        
        loss = None
        if labels is not None:
            # Calculate cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # Mamba doesn't use key-value cache
        )
    
    def generate(
        self, 
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> torch.LongTensor:
        """Generate text using the swarm model"""
        
        try:
            if hasattr(self.swarm_engine, 'generate'):
                return self.swarm_engine.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    **kwargs
                )
            else:
                # Manual generation loop
                return self._manual_generate(input_ids, max_length, temperature, top_p, do_sample)
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return input with some random tokens as fallback
            batch_size, seq_len = input_ids.shape
            new_tokens = torch.randint(0, self.config.vocab_size, (batch_size, max_length - seq_len))
            return torch.cat([input_ids, new_tokens], dim=1)
    
    def _manual_generate(self, input_ids, max_length, temperature, top_p, do_sample):
        """Manual generation when swarm engine doesn't have generate method"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(input_ids)
                logits = outputs.logits[:, -1, :] / temperature
                
                if do_sample:
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def set_active_encoders(self, num_encoders: int):
        """Set the number of active encoders"""
        if hasattr(self.swarm_engine, 'set_active_encoders'):
            self.swarm_engine.set_active_encoders(num_encoders)
            self.num_active_encoders = num_encoders
        else:
            self.num_active_encoders = min(num_encoders, self.config.max_mamba_encoders)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        """Load model from pretrained weights"""
        try:
            return super().from_pretrained(model_name_or_path, *model_args, **kwargs)
        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e}")
            # Create with default config if loading fails
            config = MambaSwarmConfig()
            return cls(config)
    
    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
