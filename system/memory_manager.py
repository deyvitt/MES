"""
Memory Manager for Mamba Swarm
Handles memory optimization, caching, and distributed memory management
"""

import torch
import torch.nn as nn
import gc
import psutil
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
import logging

@dataclass
class MemoryStats:
    total_memory: float
    used_memory: float
    free_memory: float
    gpu_memory: float
    gpu_free: float
    cache_size: float

class LRUCache:
    """Least Recently Used cache for model states and activations"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: torch.Tensor):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                old_value = self.cache.pop(oldest_key)
                del old_value
            
            self.cache[key] = value.clone() if isinstance(value, torch.Tensor) else value
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            gc.collect()

class GradientAccumulator:
    """Manages gradient accumulation across multiple steps"""
    
    def __init__(self, accumulation_steps: int = 8):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_gradients = {}
    
    def accumulate(self, model: nn.Module):
        """Accumulate gradients from current backward pass"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self.accumulated_gradients:
                    self.accumulated_gradients[name] = param.grad.clone()
                else:
                    self.accumulated_gradients[name] += param.grad
        
        self.current_step += 1
    
    def should_update(self) -> bool:
        """Check if we should perform optimizer step"""
        return self.current_step % self.accumulation_steps == 0
    
    def get_averaged_gradients(self) -> Dict[str, torch.Tensor]:
        """Get accumulated gradients averaged over accumulation steps"""
        averaged = {}
        for name, grad in self.accumulated_gradients.items():
            averaged[name] = grad / self.accumulation_steps
        return averaged
    
    def reset(self):
        """Reset accumulator"""
        self.accumulated_gradients.clear()
        self.current_step = 0

class MemoryManager:
    """Comprehensive memory management for Mamba Swarm"""
    
    def __init__(self, 
                 max_cache_size: int = 2000,
                 gradient_accumulation_steps: int = 8,
                 auto_cleanup: bool = True,
                 memory_threshold: float = 0.85):
        
        self.logger = logging.getLogger(__name__)
        self.max_cache_size = max_cache_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.auto_cleanup = auto_cleanup
        self.memory_threshold = memory_threshold
        
        # Initialize components
        self.activation_cache = LRUCache(max_cache_size)
        self.state_cache = LRUCache(max_cache_size // 2)
        self.gradient_accumulator = GradientAccumulator(gradient_accumulation_steps)
        
        # Memory tracking
        self.peak_memory_usage = 0.0
        self.memory_history = []
        self.cleanup_threshold = memory_threshold
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_memory_optimization()
    
    def setup_memory_optimization(self):
        """Setup memory optimization settings"""
        if torch.cuda.is_available():
            # Enable memory mapping for large tensors
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.9)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # System memory
        memory = psutil.virtual_memory()
        total_memory = memory.total / (1024**3)  # GB
        used_memory = memory.used / (1024**3)
        free_memory = memory.available / (1024**3)
        
        # GPU memory
        gpu_memory = 0.0
        gpu_free = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            gpu_free = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / (1024**3)
        
        # Cache size estimation
        cache_size = (len(self.activation_cache.cache) + len(self.state_cache.cache)) * 0.001  # Rough estimate
        
        stats = MemoryStats(
            total_memory=total_memory,
            used_memory=used_memory,
            free_memory=free_memory,
            gpu_memory=gpu_memory,
            gpu_free=gpu_free,
            cache_size=cache_size
        )
        
        # Update peak usage
        current_usage = used_memory + gpu_memory
        if current_usage > self.peak_memory_usage:
            self.peak_memory_usage = current_usage
        
        return stats
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        stats = self.get_memory_stats()
        memory_usage_ratio = stats.used_memory / stats.total_memory
        
        if torch.cuda.is_available():
            gpu_usage_ratio = stats.gpu_memory / (stats.gpu_memory + stats.gpu_free + 1e-6)
            return memory_usage_ratio > self.cleanup_threshold or gpu_usage_ratio > self.cleanup_threshold
        
        return memory_usage_ratio > self.cleanup_threshold
    
    def cleanup_memory(self, aggressive: bool = False):
        """Perform memory cleanup"""
        if aggressive:
            self.activation_cache.clear()
            self.state_cache.clear()
            self.gradient_accumulator.reset()
        
        # Python garbage collection
        gc.collect()
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.logger.info(f"Memory cleanup completed. Aggressive: {aggressive}")
    
    def cache_activation(self, key: str, activation: torch.Tensor):
        """Cache activation with memory pressure check"""
        if self.auto_cleanup and self.check_memory_pressure():
            self.cleanup_memory()
        
        self.activation_cache.put(key, activation)
    
    def get_cached_activation(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve cached activation"""
        return self.activation_cache.get(key)
    
    def cache_hidden_state(self, key: str, state: torch.Tensor):
        """Cache hidden state"""
        self.state_cache.put(key, state)
    
    def get_cached_state(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve cached hidden state"""
        return self.state_cache.get(key)
    
    def manage_gradient_accumulation(self, model: nn.Module) -> bool:
        """Manage gradient accumulation and return if optimizer step should be taken"""
        self.gradient_accumulator.accumulate(model)
        
        if self.gradient_accumulator.should_update():
            # Apply accumulated gradients
            averaged_grads = self.gradient_accumulator.get_averaged_gradients()
            
            for name, param in model.named_parameters():
                if name in averaged_grads:
                    param.grad = averaged_grads[name]
            
            self.gradient_accumulator.reset()
            return True
        
        return False
    
    def optimize_model_memory(self, model: nn.Module):
        """Optimize model memory usage"""
        # Enable gradient checkpointing for large models
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
        
        # Convert to half precision if possible
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
            model = model.half()
        
        return model
    
    def create_memory_efficient_dataloader(self, dataset, batch_size: int, **kwargs):
        """Create memory-efficient dataloader"""
        # Adjust batch size based on available memory
        stats = self.get_memory_stats()
        
        if stats.free_memory < 2.0:  # Less than 2GB free
            batch_size = max(1, batch_size // 2)
            self.logger.warning(f"Reduced batch size to {batch_size} due to low memory")
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=min(4, psutil.cpu_count()),
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
            **kwargs
        )
    
    def monitor_memory_usage(self):
        """Monitor and log memory usage"""
        stats = self.get_memory_stats()
        self.memory_history.append({
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
            'stats': stats
        })
        
        # Keep only recent history
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-50:]
        
        self.logger.debug(f"Memory - System: {stats.used_memory:.2f}GB/{stats.total_memory:.2f}GB, "
                         f"GPU: {stats.gpu_memory:.2f}GB, Cache: {stats.cache_size:.2f}GB")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report"""
        stats = self.get_memory_stats()
        
        return {
            'current_stats': stats.__dict__,
            'peak_usage': self.peak_memory_usage,
            'cache_stats': {
                'activation_cache_size': len(self.activation_cache.cache),
                'state_cache_size': len(self.state_cache.cache),
                'max_cache_size': self.max_cache_size
            },
            'gradient_accumulation': {
                'current_step': self.gradient_accumulator.current_step,
                'accumulation_steps': self.gradient_accumulation_steps,
                'accumulated_params': len(self.gradient_accumulator.accumulated_gradients)
            },
            'memory_pressure': self.check_memory_pressure(),
            'device': str(self.device)
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup_memory(aggressive=True) 