# upload_to_hf.py - Script to upload your Mamba Swarm to HuggingFace

import os
import shutil
from huggingface_hub import HfApi, upload_folder
import json

def prepare_model_repo():
    """Prepare model repository structure for HuggingFace"""
    
    # Create required files for HuggingFace model
    model_files = {
        "README.md": create_model_readme(),
        "config.json": create_model_config(),
        "requirements.txt": create_requirements(),
        "modeling_mamba_swarm.py": create_modeling_file()
    }
    
    # Create model repo directory
    os.makedirs("hf_model_repo", exist_ok=True)
    
    # Copy your mamba_swarm code
    shutil.copytree("mamba_swarm", "hf_model_repo/mamba_swarm", dirs_exist_ok=True)
    
    # Create HuggingFace specific files
    for filename, content in model_files.items():
        with open(f"hf_model_repo/{filename}", "w") as f:
            f.write(content)
    
    print("Model repository prepared!")

def create_model_readme():
    return """---
license: apache-2.0
language: 
- en
pipeline_tag: text-generation
tags:
- mamba
- swarm
- routing
- language-model
---

# Mamba Swarm: Dynamic Routing Language Model

A novel architecture combining 100 specialized Mamba encoders with dynamic routing and aggregation for efficient language modeling.

## Architecture

- **100 Mamba Encoders**: Specialized domain experts
- **Dynamic Router**: Selects relevant encoders per input
- **Aggregation Layer**: Combines encoder outputs
- **Mamba Decoder**: Generates final responses

## Usage

```python
from transformers import AutoModel, AutoTokenizer
from mamba_swarm import MambaSwarmEngine

# Load the model
model = MambaSwarmEngine.from_pretrained("your-username/mamba-swarm-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/mamba-swarm-model")

# Generate text
input_text = "Explain quantum computing"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training

This model uses a three-phase training approach:
1. Collective pre-training on general data
2. Domain specialization for encoder groups  
3. End-to-end coordination training

## Performance

- **Parameters**: ~7B total (100 × 70M encoders)
- **Domains**: Medical, Legal, Code, Science, General
- **Routing Efficiency**: Only 10-20% of encoders active per query

## Citation

```
@misc{mamba-swarm-2025,
  title={Mamba Swarm: Dynamic Routing for Efficient Language Modeling},
  author={Your Name},
  year={2025}
}
```
"""

def create_model_config():
    config = {
        "model_type": "mamba_swarm",
        "architectures": ["MambaSwarmForCausalLM"],
        "num_encoders": 100,
        "encoder_config": {
            "d_model": 768,
            "n_layer": 24,
            "vocab_size": 50280,
            "ssm_cfg": {},
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True
        },
        "router_config": {
            "top_k": 10,
            "routing_strategy": "content_based"
        },
        "aggregator_config": {
            "method": "weighted_sum",
            "attention_heads": 8
        },
        "torch_dtype": "float16",
        "use_cache": True
    }
    return json.dumps(config, indent=2)

def create_requirements():
    return """torch>=2.0.0
transformers>=4.35.0
mamba-ssm>=1.2.0
causal-conv1d>=1.2.0
numpy>=1.21.0
scipy>=1.7.0
triton>=2.0.0
einops>=0.6.1
packaging>=20.0
"""

def create_modeling_file():
    return """# modeling_mamba_swarm.py - HuggingFace integration

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn

class MambaSwarmConfig(PretrainedConfig):
    model_type = "mamba_swarm"
    
    def __init__(
        self,
        num_encoders=100,
        encoder_config=None,
        router_config=None,
        aggregator_config=None,
        **kwargs
    ):
        self.num_encoders = num_encoders
        self.encoder_config = encoder_config or {}
        self.router_config = router_config or {}
        self.aggregator_config = aggregator_config or {}
        super().__init__(**kwargs)

class MambaSwarmForCausalLM(PreTrainedModel):
    config_class = MambaSwarmConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Import your actual implementation
        from mamba_swarm.system.swarm_engine import MambaSwarmEngine
        
        self.swarm_engine = MambaSwarmEngine(config)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # Your forward pass implementation
        outputs = self.swarm_engine(input_ids, attention_mask)
        
        loss = None
        if labels is not None:
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
        )
    
    def generate(self, *args, **kwargs):
        return self.swarm_engine.generate(*args, **kwargs)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        # Custom loading logic if needed
        return super().from_pretrained(model_name_or_path, *model_args, **kwargs)
"""

def upload_model():
    """Upload model code to HuggingFace"""
    api = HfApi()
    
    # Upload model repository
    upload_folder(
        folder_path="hf_model_repo",
        repo_id="your-username/mamba-swarm-model",  # Replace with your username
        repo_type="model",
        commit_message="Initial upload of Mamba Swarm model"
    )
    
    print("Model uploaded successfully!")

def upload_weights():
    """Upload model weights separately"""
    # This assumes you have trained weights in checkpoints/
    api = HfApi()
    
    upload_folder(
        folder_path="checkpoints",
        repo_id="your-username/mamba-swarm-weights",  # Replace with your username
        repo_type="model", 
        commit_message="Upload trained model weights"
    )
    
    print("Weights uploaded successfully!")

if __name__ == "__main__":
    prepare_model_repo()
    upload_model()
    # upload_weights()  # Uncomment when you have trained weights