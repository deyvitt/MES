from transformers import PretrainedConfig

class MambaSwarmConfig(PretrainedConfig):
    model_type = "mamba_swarm"
    
    def __init__(
        self,
        num_mamba_encoders=5,
        max_mamba_encoders=1000,
        d_model=768,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        vocab_size=50257,
        max_sequence_length=2048,
        pad_token_id=50256,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=False,
        use_cache=True,
        gating_config=None,
        routing_config=None,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        
        self.num_mamba_encoders = num_mamba_encoders
        self.max_mamba_encoders = max_mamba_encoders
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.use_cache = use_cache
        
        # Default gating configuration
        if gating_config is None:
            gating_config = {
                "gating_type": "learned",
                "top_k": 2,
                "load_balancing_loss_coef": 0.01
            }
        self.gating_config = gating_config
        
        # Default routing configuration
        if routing_config is None:
            routing_config = {
                "routing_strategy": "dynamic",
                "aggregation_method": "weighted_average"
            }
        self.routing_config = routing_config 