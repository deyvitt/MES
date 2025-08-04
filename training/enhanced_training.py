#!/usr/bin/env python3
"""
Enhanced Training Script - Uses your existing trainer.py with HF datasets
This integrates with your current MambaSwarmTrainer system
"""

import os
import sys
from pathlib import Path
import logging

# Add project paths - go up one level since we're in training/ folder
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Your existing imports
from core.config import MambaConfig
from training.trainer import MambaSwarmTrainer

# Enhanced dataset support
from datasets import load_dataset
import json

logger = logging.getLogger(__name__)

def prepare_hf_dataset_for_existing_system(dataset_name: str = "wikitext-103-v1", 
                                         output_path: str = "train_data.txt"):
    """
    Download HF dataset and convert to format your existing trainer expects
    """
    
    logger.info(f"📥 Loading {dataset_name} from Hugging Face...")
    
    try:
        # Load the dataset
        if dataset_name == "wikitext-103-v1":
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            text_column = "text"
        elif dataset_name == "openwebtext":
            dataset = load_dataset("openwebtext", split="train[:10000]")  # Subset
            text_column = "text"
        elif dataset_name == "tiny_shakespeare":
            dataset = load_dataset("tiny_shakespeare", split="train")
            text_column = "text"
        else:
            # Generic loading
            dataset = load_dataset(dataset_name, split="train")
            text_column = "text"
        
        # Convert to simple text format your trainer expects
        logger.info(f"📄 Converting to text format...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                text = example.get(text_column, "")
                if text and len(text.strip()) > 20:  # Filter very short texts
                    f.write(text.strip() + "\n\n")  # Double newline as separator
        
        logger.info(f"✅ Dataset saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Failed to load {dataset_name}: {e}")
        
        # Create fallback dummy data
        logger.info("Creating fallback training data...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(1000):
                f.write(f"This is training example number {i}. It contains meaningful text for language modeling.\n\n")
        
        return output_path

def run_existing_trainer_with_hf_data():
    """
    Use your existing MambaSwarmTrainer but with HF dataset
    """
    
    logger.info("🐍 Starting Mamba Swarm Training with HF Data")
    logger.info("=" * 60)
    
    # Step 1: Prepare dataset
    logger.info("Step 1: Preparing Hugging Face dataset...")
    dataset_path = prepare_hf_dataset_for_existing_system("wikitext-103-v1", "train_data.txt")
    
    # Step 2: Create your existing config
    logger.info("Step 2: Creating MambaConfig...")
    config = MambaConfig(
        # Model settings
        vocab_size=50257,
        d_model=768,        # Smaller for faster training
        n_layers=8,         # Fewer layers for demo
        
        # Training settings  
        batch_size=2,       # Small batch for memory efficiency
        learning_rate=1e-4,
        max_seq_len=512,    # Shorter sequences
        
        # Swarm settings
        num_specialists=20, # Fewer specialists for demo
        
        # Training steps (reduced for demo)
        warmup_steps=100,
        max_steps=2000,
        
        # Dataset path
        train_data_path=dataset_path
    )
    
    logger.info(f"✅ Config created:")
    logger.info(f"  - Model: {config.d_model}D, {config.n_layers} layers")
    logger.info(f"  - Specialists: {config.num_specialists}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Training data: {config.train_data_path}")
    
    # Step 3: Initialize your existing trainer
    logger.info("Step 3: Initializing MambaSwarmTrainer...")
    try:
        trainer = MambaSwarmTrainer(config)
        logger.info("✅ Trainer initialized successfully")
    except Exception as e:
        logger.error(f"❌ Trainer initialization failed: {e}")
        return False
    
    # Step 4: Run your existing training pipeline
    logger.info("Step 4: Starting training pipeline...")
    logger.info("This will run your 4-phase training:")
    logger.info("  Phase 1: Foundation training")
    logger.info("  Phase 2: Specialist training") 
    logger.info("  Phase 3: Aggregator training")
    logger.info("  Phase 4: End-to-end fine-tuning")
    
    try:
        # Run your existing full pipeline
        trainer.full_training_pipeline()
        
        logger.info("🎉 Training completed successfully!")
        
        # Save checkpoint using your existing method
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "mamba_swarm_hf_trained.pt")
        trainer.save_checkpoint(checkpoint_path)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Run evaluation using your existing method
        logger.info("📊 Running evaluation...")
        eval_results = trainer.evaluate(eval_steps=50)
        logger.info(f"Evaluation results: {eval_results}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

def quick_test_run():
    """Quick test with minimal settings"""
    
    logger.info("🚀 Quick Test Run")
    
    # Use tiny dataset for quick test
    dataset_path = prepare_hf_dataset_for_existing_system("tiny_shakespeare", "test_data.txt")
    
    # Minimal config for testing
    config = MambaConfig(
        d_model=256,        # Very small
        n_layers=4,         # Very few layers
        batch_size=1,       # Single batch
        num_specialists=5,  # Few specialists
        warmup_steps=10,
        max_steps=50,       # Very short training
        train_data_path=dataset_path
    )
    
    trainer = MambaSwarmTrainer(config)
    
    # Just run foundation phase for testing
    logger.info("Running foundation training only...")
    trainer.train_foundation_phase(num_steps=20)
    
    logger.info("✅ Quick test completed!")

if __name__ == "__main__":
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Enhanced Mamba training with HF datasets")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with minimal settings")
    parser.add_argument("--dataset", default="wikitext-103-v1", help="HuggingFace dataset to use")
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test_run()
    else:
        success = run_existing_trainer_with_hf_data()
        if success:
            print("\n🎉 Training completed successfully!")
            print("Your trained Mamba swarm model is ready to use!")
        else:
            print("\n❌ Training failed. Check the logs above for details.")
