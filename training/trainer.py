# =============================================================================
# training/trainer.py
# =============================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import time
import logging
from pathlib import Path

from core.config import MambaConfig
from routing.tlm_manager import TLMManager
from routing.aggregator import AttentionAggregator
from training.optimizer import MambaOptimizer
from training.loss import MambaLoss
from training.data_loader import create_data_loaders
from core.tokenizer import MambaTokenizer
from core.preprocess import TextPreprocessor

class MambaSwarmTrainer:
    """Multi-phase trainer for Mamba swarm architecture"""
    
    def __init__(self, config: MambaConfig):
        self.config = config
        self.device = config.device
        
        # Initialize components
        self.tokenizer = MambaTokenizer(config)
        self.preprocessor = TextPreprocessor(config)
        
        # Initialize TLM manager and aggregator
        self.tlm_manager = TLMManager(config)
        self.aggregator = AttentionAggregator(config)
        self.aggregator.to(self.device)
        
        # Initialize loss function
        self.loss_fn = MambaLoss(config, config.vocab_size)
        
        # Create data loaders
        self.data_loaders = create_data_loaders(config, self.tokenizer, self.preprocessor)
        
        # Training state
        self.global_step = 0
        self.phase = "foundation"  # foundation, specialists, aggregator, end_to_end
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup training logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_foundation_phase(self, num_steps: int = 10000):
        """Phase 1: Train shared foundation weights"""
        self.logger.info("Starting foundation training phase...")
        self.phase = "foundation"
        
        # Get a reference specialist for foundation training
        reference_specialist = list(self.tlm_manager.specialists.values())[0]
        optimizer = MambaOptimizer(reference_specialist.model, self.config)
        
        reference_specialist.model.train()
        
        for step in range(num_steps):
            batch = next(iter(self.data_loaders['main']))
            
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass
            logits, loss = reference_specialist.model(input_ids, target_ids)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            lr = optimizer.step()
            
            self.global_step += 1
            
            if step % 100 == 0:
                self.logger.info(f"Foundation step {step}, loss: {loss.item():.4f}, lr: {lr:.6f}")
        
        # Copy foundation weights to all specialists
        self._copy_foundation_weights(reference_specialist)
        
        self.logger.info("Foundation training phase completed!")
    
    def _copy_foundation_weights(self, reference_specialist):
        """Copy foundation weights to all specialists"""
        reference_state = reference_specialist.model.state_dict()
        
        for specialist in self.tlm_manager.specialists.values():
            if specialist != reference_specialist:
                # Copy shared layers (first half of the model)
                specialist_state = specialist.model.state_dict()
                
                for name, param in reference_state.items():
                    if 'layers.' in name:
                        # Extract layer number
                        layer_num = int(name.split('.')[1])
                        if layer_num < self.config.n_layers // 2:  # Share first half
                            specialist_state[name] = param.clone()
                    elif 'embedding' in name:  # Share embeddings
                        specialist_state[name] = param.clone()
                
                specialist.model.load_state_dict(specialist_state)
    
    def train_specialists_phase(self, num_steps: int = 5000):
        """Phase 2: Train domain specialists in parallel"""
        self.logger.info("Starting specialist training phase...")
        self.phase = "specialists"
        
        # Create optimizers for each specialist
        specialist_optimizers = {}
        for specialist_id, specialist in self.tlm_manager.specialists.items():
            specialist_optimizers[specialist_id] = MambaOptimizer(
                specialist.model, self.config
            )
            specialist.model.train()
        
        # Train specialists in parallel (simplified - could use actual parallel training)
        for step in range(num_steps):
            total_loss = 0.0
            
            # Train each specialist on its domain data
            for specialist_id in range(min(10, self.config.num_specialists)):  # Limit for demo
                if specialist_id in self.data_loaders['domains']:
                    try:
                        batch = next(iter(self.data_loaders['domains'][specialist_id]))
                        
                        # Move to device
                        input_ids = batch['input_ids'].to(self.device)
                        target_ids = batch['target_ids'].to(self.device)
                        
                        # Get specialist and optimizer
                        specialist = self.tlm_manager.specialists[specialist_id]
                        optimizer = specialist_optimizers[specialist_id]
                        
                        # Forward pass
                        logits, loss = specialist.model(input_ids, target_ids)
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        
                    except Exception as e:
                        self.logger.warning(f"Error training specialist {specialist_id}: {e}")
                        continue
            
            self.global_step += 1
            
            if step % 100 == 0:
                avg_loss = total_loss / min(10, self.config.num_specialists)
                self.logger.info(f"Specialists step {step}, avg loss: {avg_loss:.4f}")
        
        self.logger.info("Specialist training phase completed!")
    
    def train_aggregator_phase(self, num_steps: int = 3000):
        """Phase 3: Train aggregator to combine specialist outputs"""
        self.logger.info("Starting aggregator training phase...")
        self.phase = "aggregator"
        
        # Freeze specialist models
        for specialist in self.tlm_manager.specialists.values():
            specialist.model.eval()
            for param in specialist.model.parameters():
                param.requires_grad = False
        
        # Create optimizer for aggregator
        aggregator_optimizer = MambaOptimizer(self.aggregator, self.config)
        self.aggregator.train()
        
        for step in range(num_steps):
            try:
                batch = next(iter(self.data_loaders['main']))
                
                # Simulate specialist outputs (simplified for demo)
                specialist_outputs = self._simulate_specialist_outputs(batch)
                
                # Get target text for comparison
                target_ids = batch['target_ids'].to(self.device)
                
                # Forward pass through aggregator
                logits = self.aggregator(specialist_outputs)
                
                # Compute loss
                loss_dict = self.loss_fn(logits, target_ids)
                loss = loss_dict['total_loss']
                
                # Backward pass
                aggregator_optimizer.zero_grad()
                loss.backward()
                aggregator_optimizer.step()
                
                self.global_step += 1
                
                if step % 100 == 0:
                    self.logger.info(f"Aggregator step {step}, loss: {loss.item():.4f}")
                    
            except Exception as e:
                self.logger.warning(f"Error in aggregator training step {step}: {e}")
                continue
        
        self.logger.info("Aggregator training phase completed!")
    
    def _simulate_specialist_outputs(self, batch) -> Dict[int, List[Dict]]:
        """Simulate specialist outputs for aggregator training"""
        # This is a simplified simulation - in real training, you'd run
        # the text through the router and specialists
        
        input_ids = batch['input_ids'].to(self.device)
        
        # Simulate 3 chunks with 2-3 specialists each
        specialist_outputs = {}
        
        for chunk_id in range(3):
            chunk_results = []
            
            # Simulate 2-3 specialists working on this chunk
            for i in range(2 + chunk_id % 2):
                specialist_id = (chunk_id * 3 + i) % self.config.num_specialists
                
                if specialist_id in self.tlm_manager.specialists:
                    specialist = self.tlm_manager.specialists[specialist_id]
                    
                    # Get encoding from specialist
                    with torch.no_grad():
                        encoding = specialist.encode(input_ids[:1])  # Single sample
                    
                    chunk_results.append({
                        'chunk_id': chunk_id,
                        'specialist_id': specialist_id,
                        'confidence': 0.8 + 0.2 * torch.rand(1).item(),
                        'encoding': encoding[0],  # Remove batch dim
                        'domain': f'domain_{specialist_id}'
                    })
            
            specialist_outputs[chunk_id] = chunk_results
        
        return specialist_outputs
    
    def train_end_to_end_phase(self, num_steps: int = 2000):
        """Phase 4: End-to-end fine-tuning of the entire system"""
        self.logger.info("Starting end-to-end training phase...")
        self.phase = "end_to_end"
        
        # Unfreeze all parameters
        for specialist in self.tlm_manager.specialists.values():
            specialist.model.train()
            for param in specialist.model.parameters():
                param.requires_grad = True
        
        self.aggregator.train()
        
        # Create system-wide optimizer with lower learning rate
        all_params = []
        
        # Add specialist parameters
        for specialist in self.tlm_manager.specialists.values():
            all_params.extend(specialist.model.parameters())
        
        # Add aggregator parameters
        all_params.extend(self.aggregator.parameters())
        
        # Create optimizer with reduced learning rate
        end_to_end_config = self.config
        end_to_end_config.learning_rate = self.config.learning_rate * 0.1
        
        system_optimizer = torch.optim.AdamW(
            all_params,
            lr=end_to_end_config.learning_rate,
            weight_decay=end_to_end_config.weight_decay
        )
        
        for step in range(num_steps):
            try:
                batch = next(iter(self.data_loaders['main']))
                
                # Full system forward pass (simplified)
                specialist_outputs = self._simulate_specialist_outputs(batch)
                logits = self.aggregator(specialist_outputs)
                
                # Compute loss
                target_ids = batch['target_ids'].to(self.device)
                loss_dict = self.loss_fn(logits, target_ids)
                loss = loss_dict['total_loss']
                
                # Backward pass
                system_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                system_optimizer.step()
                
                self.global_step += 1
                
                if step % 100 == 0:
                    self.logger.info(f"End-to-end step {step}, loss: {loss.item():.4f}")
                    
            except Exception as e:
                self.logger.warning(f"Error in end-to-end training step {step}: {e}")
                continue
        
        self.logger.info("End-to-end training phase completed!")
    
    def full_training_pipeline(self):
        """Run the complete 4-phase training pipeline"""
        self.logger.info("Starting full Mamba swarm training pipeline...")
        
        start_time = time.time()
        
        try:
            # Phase 1: Foundation training
            self.train_foundation_phase(num_steps=1000)  # Reduced for demo
            
            # Phase 2: Specialist training
            self.train_specialists_phase(num_steps=500)  # Reduced for demo
            
            # Phase 3: Aggregator training
            self.train_aggregator_phase(num_steps=300)   # Reduced for demo
            
            # Phase 4: End-to-end fine-tuning
            self.train_end_to_end_phase(num_steps=200)   # Reduced for demo
            
            total_time = time.time() - start_time
            self.logger.info(f"Training completed in {total_time:.2f} seconds!")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save training checkpoint"""
        checkpoint = {
            'global_step': self.global_step,
            'phase': self.phase,
            'config': self.config.__dict__,
            'aggregator_state': self.aggregator.state_dict(),
            'specialist_states': {}
        }
        
        # Save specialist states
        for specialist_id, specialist in self.tlm_manager.specialists.items():
            checkpoint['specialist_states'][specialist_id] = specialist.model.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.global_step = checkpoint['global_step']
        self.phase = checkpoint['phase']
        
        # Load aggregator state
        self.aggregator.load_state_dict(checkpoint['aggregator_state'])
        
        # Load specialist states
        for specialist_id, state_dict in checkpoint['specialist_states'].items():
            if specialist_id in self.tlm_manager.specialists:
                self.tlm_manager.specialists[specialist_id].model.load_state_dict(state_dict)
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def evaluate(self, eval_steps: int = 100) -> Dict[str, float]:
        """Evaluate the trained model"""
        self.logger.info("Starting evaluation...")
        
        # Set models to eval mode
        for specialist in self.tlm_manager.specialists.values():
            specialist.model.eval()
        self.aggregator.eval()
        
        total_loss = 0.0
        num_steps = 0
        
        with torch.no_grad():
            for step in range(eval_steps):
                try:
                    batch = next(iter(self.data_loaders['main']))
                    
                    # Forward pass
                    specialist_outputs = self._simulate_specialist_outputs(batch)
                    logits = self.aggregator(specialist_outputs)
                    
                    # Compute loss
                    target_ids = batch['target_ids'].to(self.device)
                    loss_dict = self.loss_fn(logits, target_ids)
                    
                    total_loss += loss_dict['total_loss'].item()
                    num_steps += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error in evaluation step {step}: {e}")
                    continue
        
        avg_loss = total_loss / max(num_steps, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        results = {
            'eval_loss': avg_loss,
            'perplexity': perplexity,
            'num_steps': num_steps
        }
        
        self.logger.info(f"Evaluation results: {results}")
        return results