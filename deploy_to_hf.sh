#!/bin/bash
# deploy_to_hf.sh - Complete deployment script

echo "🚀 Deploying Mamba Swarm to HuggingFace..."

# Set your HuggingFace username
HF_USERNAME="your-username"  # Replace with your actual username

# Step 1: Create repositories on HuggingFace
echo "📦 Creating repositories..."
huggingface-cli repo create mamba-swarm-model --type model
huggingface-cli repo create mamba-swarm-weights --type model  
huggingface-cli repo create mamba-swarm-demo --type space --space_sdk gradio

# Step 2: Clone repositories locally
echo "📁 Cloning repositories..."
mkdir -p hf_repos
cd hf_repos

git clone https://huggingface.co/$HF_USERNAME/mamba-swarm-model
git clone https://huggingface.co/$HF_USERNAME/mamba-swarm-weights
git clone https://huggingface.co/$HF_USERNAME/mamba-swarm-demo

# Step 3: Prepare model repository
echo "🔧 Preparing model code..."
cd mamba-swarm-model

# Copy your mamba_swarm code
cp -r ../../mamba_swarm .

# Create README.md
cat > README.md << 'EOF'
---
license: apache-2.0
language: 
- en
pipeline_tag: text-generation
tags:
- mamba
- swarm
- routing
- language-model
library_name: transformers
---

# Mamba Swarm: Dynamic Routing Language Model

A novel architecture combining 100 specialized Mamba encoders with dynamic routing and aggregation for efficient language modeling.

## Quick Start

```python
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
model = AutoModel.from_pretrained("$HF_USERNAME/mamba-swarm-model")
tokenizer = AutoTokenizer.from_pretrained("$HF_USERNAME/mamba-swarm-model")

# Generate text
input_text = "Explain quantum computing"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Architecture

- **100 Mamba Encoders**: Domain-specialized experts
- **Dynamic Router**: Content-aware encoder selection  
- **Aggregation Layer**: Intelligent output combination
- **Mamba Decoder**: Coherent response generation

## Demo

Try the interactive demo: [Mamba Swarm Demo](https://huggingface.co/spaces/$HF_USERNAME/mamba-swarm-demo)
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.35.0
mamba-ssm>=1.2.0
causal-conv1d>=1.2.0
numpy>=1.21.0
scipy>=1.7.0
triton>=2.0.0
einops>=0.6.1
packaging>=20.0
accelerate>=0.20.0
EOF

# Create config.json
cat > config.json << 'EOF'
{
  "model_type": "mamba_swarm",
  "architectures": ["MambaSwarmForCausalLM"],
  "num_encoders": 100,
  "encoder_config": {
    "d_model": 768,
    "n_layer": 24,
    "vocab_size": 50280,
    "ssm_cfg": {},
    "rms_norm": true,
    "residual_in_fp32": true,
    "fused_add_norm": true
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
  "use_cache": true
}
EOF

# Commit and push model code
git add .
git commit -m "Initial upload: Mamba Swarm model code"
git push

echo "✅ Model code uploaded!"

# Step 4: Prepare Gradio app
echo "🎨 Preparing Gradio demo..."
cd ../mamba-swarm-demo

# Copy the app.py file we created
cp ../../gradio_app.py app.py

# Update the model name in app.py
sed -i "s/your-username/$HF_USERNAME/g" app.py

# Create requirements.txt for the Space
cat > requirements.txt << 'EOF'
gradio>=4.0.0
torch>=2.0.0
transformers>=4.35.0
numpy>=1.21.0
mamba-ssm>=1.2.0
causal-conv1d>=1.2.0
EOF

# Create README.md for the Space
cat > README.md << 'EOF'
---
title: Mamba Swarm Demo
emoji: 🐍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Mamba Swarm Interactive Demo

Experience the power of 100 specialized Mamba encoders with intelligent routing!

This demo showcases how our Mamba Swarm model dynamically selects the most relevant encoders for different types of queries, providing specialized responses across various domains.

## Features

- **Dynamic Routing**: Watch as the model selects optimal encoders
- **Domain Specialization**: See how different domains are handled
- **Interactive Interface**: Experiment with different parameters
- **Real-time Visualization**: View routing decisions and confidence scores

## Architecture

The Mamba Swarm consists of:
- 100 specialized Mamba encoders
- Intelligent content-based routing
- Advanced aggregation mechanisms
- Optimized inference pipeline

Try it out with different types of questions to see the routing in action!
EOF

# Commit and push Gradio app
git add .
git commit -m "Initial upload: Mamba Swarm Gradio demo"
git push

echo "✅ Gradio demo uploaded!"

# Step 5: Instructions for weights (when available)
echo "📋 Next steps for model weights:"
echo ""
echo "When you have trained model weights, upload them with:"
echo "cd hf_repos/mamba-swarm-weights"
echo "# Copy your checkpoint files here"
echo "git add ."
echo "git commit -m 'Upload trained model weights'"
echo "git push"
echo ""
echo "🎉 Deployment complete!"
echo ""
echo "Your repositories:"
echo "- Model: https://huggingface.co/$HF_USERNAME/mamba-swarm-model"
echo "- Weights: https://huggingface.co/$HF_USERNAME/mamba-swarm-weights"  
echo "- Demo: https://huggingface.co/$HF_USERNAME/mamba-swarm-demo"
echo ""
echo "The Gradio demo will be available at:"
echo "https://huggingface.co/spaces/$HF_USERNAME/mamba-swarm-demo" 