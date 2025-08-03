#!/usr/bin/env python3
"""
Mamba GPT-3 Competitor - Phase 1 Starter
A cheap but accurate language model using Mamba architecture
"""

import torch
import torch.nn.functional as F
from transformers import (
    MambaConfig, 
    MambaForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import wandb
import os
from datetime import datetime

class MambaGPT3Config:
    """Configuration for our GPT-3 competitor"""
    
    # Model Architecture
    VOCAB_SIZE = 50257  # GPT-2 tokenizer vocab size
    HIDDEN_SIZE = 768   # Start small for proof of concept
    NUM_LAYERS = 12     # Fewer layers than GPT-3's 96
    STATE_SIZE = 16     # Mamba's state size
    
    # Training
    BATCH_SIZE = 4      # Fit in 6GB GPU
    GRADIENT_ACCUMULATION = 4  # Effective batch size = 16
    LEARNING_RATE = 5e-4
    MAX_STEPS = 10000   # Quick proof of concept
    WARMUP_STEPS = 1000
    
    # Data
    MAX_LENGTH = 1024   # Context length
    DATASET_NAME = "openwebtext"  # High quality text data
    
    # Logging
    EVAL_STEPS = 500
    SAVE_STEPS = 1000
    LOGGING_STEPS = 100

def create_mamba_model(config_class=MambaGPT3Config):
    """Create a Mamba model optimized for language modeling"""
    
    config = MambaConfig(
        vocab_size=config_class.VOCAB_SIZE,
        hidden_size=config_class.HIDDEN_SIZE,
        state_size=config_class.STATE_SIZE,
        num_hidden_layers=config_class.NUM_LAYERS,
        expand=2,  # Expansion factor for feed-forward
        conv_kernel=4,  # Convolution kernel size
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",  # SiLU activation
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_scale=1.0,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_init_scheme="random",
        time_step_floor=1e-4,
        rescale_prenorm_residual=False,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
    )
    
    model = MambaForCausalLM(config)
    
    # Calculate and print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 Created Mamba model with {num_params:,} parameters")
    print(f"📊 Model size: ~{num_params / 1e6:.1f}M parameters")
    
    return model, config

def load_tokenizer():
    """Load GPT-2 tokenizer (compatible with our vocab size)"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def prepare_dataset(tokenizer, config_class=MambaGPT3Config):
    """Load and prepare training dataset"""
    print("📚 Loading dataset...")
    
    # Load a subset for quick testing
    dataset = load_dataset(
        config_class.DATASET_NAME, 
        split="train[:1%]",  # Use 1% for quick start
        streaming=False
    )
    
    def tokenize_function(examples):
        # Tokenize and chunk into max_length sequences
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=config_class.MAX_LENGTH,
            return_overflowing_tokens=True,
        )
        return tokenized
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    print(f"📊 Dataset size: {len(tokenized_dataset):,} examples")
    return tokenized_dataset

def setup_training(model, tokenizer, dataset, config_class=MambaGPT3Config):
    """Setup training configuration"""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./mamba_gpt3_checkpoints/{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training
        num_train_epochs=1,  # We'll use max_steps instead
        max_steps=config_class.MAX_STEPS,
        per_device_train_batch_size=config_class.BATCH_SIZE,
        gradient_accumulation_steps=config_class.GRADIENT_ACCUMULATION,
        learning_rate=config_class.LEARNING_RATE,
        warmup_steps=config_class.WARMUP_STEPS,
        
        # Optimization for 6GB GPU
        fp16=True,  # Mixed precision to save memory
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Trade compute for memory
        
        # Logging and evaluation
        logging_steps=config_class.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config_class.EVAL_STEPS,
        save_steps=config_class.SAVE_STEPS,
        save_total_limit=3,
        
        # Misc
        report_to="wandb",  # Log to Weights & Biases
        run_name=f"mamba_gpt3_{timestamp}",
        seed=42,
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling, not masked
    )
    
    # Split dataset
    train_dataset = dataset
    eval_dataset = dataset.select(range(min(1000, len(dataset))))  # Small eval set
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    return trainer

def test_generation(model, tokenizer, device):
    """Test the model's text generation capabilities"""
    model.eval()
    
    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The most important lesson I learned was",
    ]
    
    print("\n🎯 Testing text generation:")
    print("=" * 50)
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text[len(prompt):]}")
        print("-" * 30)

def main():
    """Main training pipeline"""
    print("🚀 Starting Mamba GPT-3 Competitor Training")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Initialize Weights & Biases
    wandb.init(
        project="mamba-gpt3-competitor",
        name=f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=MambaGPT3Config.__dict__
    )
    
    # Create model and tokenizer
    model, config = create_mamba_model()
    tokenizer = load_tokenizer()
    model.to(device)
    
    # Test generation before training
    print("\n🧪 Pre-training generation test:")
    test_generation(model, tokenizer, device)
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer)
    
    # Setup training
    trainer = setup_training(model, tokenizer, dataset)
    
    # Start training
    print("\n🏋️  Starting training...")
    trainer.train()
    
    # Test generation after training
    print("\n🎉 Post-training generation test:")
    test_generation(model, tokenizer, device)
    
    # Save final model
    final_model_path = "./mamba_gpt3_final"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n✅ Training complete! Model saved to {final_model_path}")
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    wandb.finish()

if __name__ == "__main__":
    main() 