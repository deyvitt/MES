#!/usr/bin/env python3
"""
Simple Mamba Model Test
Just create and test a small Mamba model
"""

import torch
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

def test_mamba_creation():
    """Test creating a small Mamba model"""
    print("🔧 Creating Mamba model...")
    
    # Small config for testing
    config = MambaConfig(
        vocab_size=1000,
        hidden_size=256,
        state_size=16,
        num_hidden_layers=4,
    )
    
    model = MambaForCausalLM(config)
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Model created with {num_params:,} parameters")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 10))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"✅ Forward pass successful, logits shape: {outputs.logits.shape}")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        input_ids = input_ids.cuda()
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"✅ GPU forward pass successful")
        print(f"📊 GPU memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    
    return model

def test_with_real_tokenizer():
    """Test with actual GPT-2 tokenizer"""
    print("\n🔤 Testing with real tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model with proper vocab size
    config = MambaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        state_size=16,
        num_hidden_layers=6,
    )
    
    model = MambaForCausalLM(config)
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Real model created with {num_params:,} parameters (~{num_params/1e6:.1f}M)")
    
    # Test text generation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"🎯 Testing generation with prompt: '{prompt}'")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"📝 Generated: {generated_text}")
    
    return model, tokenizer

if __name__ == "__main__":
    print("🧪 Simple Mamba Model Test")
    print("=" * 40)
    
    # Test 1: Basic model creation
    model1 = test_mamba_creation()
    
    # Test 2: Real tokenizer and generation
    model2, tokenizer = test_with_real_tokenizer()
    
    print("\n🎉 All tests passed! Ready for full training.")