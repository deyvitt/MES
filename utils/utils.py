# =============================================================================
# utils/utils.py - Utility Functions for Mamba Encoder Swarm Architecture
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import logging
import os
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import warnings
from functools import wraps, lru_cache
import hashlib
import pickle

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# PERFORMANCE MONITORING UTILITIES
# =============================================================================

class PerformanceMonitor:
    """Monitor and track performance metrics for the swarm architecture"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics = defaultdict(list)
        self.max_history = max_history
        self.start_times = {}
        self.counters = defaultdict(int)
        self.lock = threading.Lock()
    
    def start_timer(self, name: str) -> None:
        """Start timing an operation"""
        with self.lock:
            self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and record duration"""
        with self.lock:
            if name in self.start_times:
                duration = time.time() - self.start_times[name]
                self.record_metric(f"{name}_duration", duration)
                del self.start_times[name]
                return duration
            return 0.0
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value"""
        with self.lock:
            self.metrics[name].append({
                'value': value,
                'timestamp': time.time()
            })
            # Keep only recent history
            if len(self.metrics[name]) > self.max_history:
                self.metrics[name] = self.metrics[name][-self.max_history:]
    
    def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a counter"""
        with self.lock:
            self.counters[name] += amount
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = [m['value'] for m in self.metrics[name]]
            return {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'recent': values[-10:] if len(values) >= 10 else values
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete performance summary"""
        with self.lock:
            summary = {
                'metrics': {name: self.get_stats(name) for name in self.metrics},
                'counters': dict(self.counters),
                'active_timers': list(self.start_times.keys()),
                'timestamp': datetime.now().isoformat()
            }
            return summary

# Global performance monitor instance
perf_monitor = PerformanceMonitor()

def monitor_performance(func_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            perf_monitor.start_timer(name)
            perf_monitor.increment_counter(f"{name}_calls")
            try:
                result = func(*args, **kwargs)
                perf_monitor.increment_counter(f"{name}_success")
                return result
            except Exception as e:
                perf_monitor.increment_counter(f"{name}_errors")
                raise
            finally:
                perf_monitor.end_timer(name)
        
        return wrapper
    return decorator

# =============================================================================
# MEMORY MANAGEMENT UTILITIES
# =============================================================================

class MemoryTracker:
    """Track memory usage across the swarm system"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory information"""
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f'gpu_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i) / 1024**3,
                    'cached': torch.cuda.memory_reserved(i) / 1024**3,
                    'max_allocated': torch.cuda.max_memory_allocated(i) / 1024**3
                }
        
        return {
            'process_memory_gb': memory_info.rss / 1024**3,
            'system_memory_percent': virtual_memory.percent,
            'system_memory_available_gb': virtual_memory.available / 1024**3,
            'gpu_memory': gpu_memory
        }
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def optimize_memory():
        """Perform memory optimization"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def memory_efficient(clear_cache: bool = True):
    """Decorator for memory-efficient functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if clear_cache:
                MemoryTracker.clear_gpu_cache()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if clear_cache:
                    MemoryTracker.clear_gpu_cache()
        
        return wrapper
    return decorator

# =============================================================================
# TENSOR UTILITIES
# =============================================================================

class TensorUtils:
    """Utility functions for tensor operations"""
    
    @staticmethod
    def safe_tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Safely move tensor to device with error handling"""
        try:
            if tensor.device != device:
                return tensor.to(device)
            return tensor
        except RuntimeError as e:
            logger.warning(f"Failed to move tensor to {device}: {e}")
            return tensor
    
    @staticmethod
    def get_tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
        """Get comprehensive tensor information"""
        return {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'requires_grad': tensor.requires_grad,
            'memory_mb': tensor.numel() * tensor.element_size() / 1024**2,
            'is_contiguous': tensor.is_contiguous(),
            'stride': tensor.stride() if tensor.dim() > 0 else []
        }
    
    @staticmethod
    def batch_tensors(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        """Batch tensors with padding to same length"""
        if not tensors:
            return torch.empty(0)
        
        max_len = max(t.size(-1) for t in tensors)
        batch_size = len(tensors)
        
        if len(tensors[0].shape) == 1:
            batched = torch.full((batch_size, max_len), pad_value, dtype=tensors[0].dtype, device=tensors[0].device)
        else:
            feature_dim = tensors[0].size(-2)
            batched = torch.full((batch_size, feature_dim, max_len), pad_value, dtype=tensors[0].dtype, device=tensors[0].device)
        
        for i, tensor in enumerate(tensors):
            if len(tensor.shape) == 1:
                batched[i, :tensor.size(0)] = tensor
            else:
                batched[i, :, :tensor.size(-1)] = tensor
        
        return batched
    
    @staticmethod
    def split_tensor_by_chunks(tensor: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        """Split tensor into chunks of specified size"""
        if tensor.size(0) <= chunk_size:
            return [tensor]
        
        return [tensor[i:i + chunk_size] for i in range(0, tensor.size(0), chunk_size)]

# =============================================================================
# ROUTING UTILITIES
# =============================================================================

class RoutingUtils:
    """Utilities for encoder routing and load balancing"""
    
    @staticmethod
    def calculate_load_balance_loss(routing_weights: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Calculate load balance loss to encourage equal encoder usage"""
        # routing_weights: [batch_size, seq_len, num_encoders]
        avg_routing = routing_weights.mean(dim=[0, 1])  # [num_encoders]
        
        # Variance penalty to encourage uniform distribution
        target_load = 1.0 / routing_weights.size(-1)
        load_balance_loss = torch.var(avg_routing) / (target_load ** 2 + epsilon)
        
        return load_balance_loss
    
    @staticmethod
    def apply_top_k_routing(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply top-k routing with Gumbel softmax"""
        # Get top-k indices
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # Create mask for top-k
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, top_k_indices, 1.0)
        
        # Apply Gumbel softmax to top-k
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(top_k_logits) + 1e-8) + 1e-8)
        top_k_weights = F.softmax((top_k_logits + gumbel_noise) / 1.0, dim=-1)
        
        # Reconstruct full weights
        weights = torch.zeros_like(logits)
        weights.scatter_(-1, top_k_indices, top_k_weights)
        
        return weights, mask
    
    @staticmethod
    def entropy_regularization(routing_weights: torch.Tensor) -> torch.Tensor:
        """Add entropy regularization to encourage exploration"""
        # Avoid log(0)
        routing_weights = torch.clamp(routing_weights, min=1e-8)
        entropy = -torch.sum(routing_weights * torch.log(routing_weights), dim=-1)
        return -entropy.mean()  # Negative because we want to maximize entropy

# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

class TextUtils:
    """Utilities for text processing and analysis"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
            
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
        """Estimate number of tokens in text"""
        return max(1, int(len(text) / chars_per_token))
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection based on character patterns"""
        # This is a simplified version - for production, use langdetect library
        if not text:
            return "unknown"
        
        # Count character types
        ascii_count = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text)
        
        if total_chars == 0:
            return "unknown"
        
        ascii_ratio = ascii_count / total_chars
        
        if ascii_ratio > 0.9:
            return "en"  # Likely English
        elif ascii_ratio > 0.7:
            return "mixed"
        else:
            return "non-latin"

# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

class ConfigUtils:
    """Utilities for configuration management"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> bool:
        """Save configuration to JSON file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved configuration to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            return False
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigUtils.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """Validate configuration has required keys"""
        missing_keys = []
        
        for key in required_keys:
            if '.' in key:
                # Handle nested keys
                keys = key.split('.')
                current = config
                for k in keys:
                    if not isinstance(current, dict) or k not in current:
                        missing_keys.append(key)
                        break
                    current = current[k]
            elif key not in config:
                missing_keys.append(key)
        
        return missing_keys

# =============================================================================
# CACHING UTILITIES
# =============================================================================

class CacheManager:
    """Intelligent caching for model outputs and computations"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        return hashlib.md5(pickle.dumps(key_data)).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.cache[key]['timestamp'] > self.ttl_seconds:
                self._remove_key(key)
                return None
            
            self.access_times[key] = time.time()
            return self.cache[key]['value']
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache"""
        with self.lock:
            # Clean up if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            self.access_times[key] = time.time()
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
    
    def clear(self) -> None:
        """Clear all cached items"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_ratio': getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1),
                'ttl_seconds': self.ttl_seconds
            }

# Global cache manager
cache_manager = CacheManager()

def cached(ttl_seconds: int = 3600):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache_manager._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache_manager.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator

# =============================================================================
# DEBUGGING AND LOGGING UTILITIES
# =============================================================================

class DebugUtils:
    """Utilities for debugging the swarm architecture"""
    
    @staticmethod
    def log_tensor_stats(tensor: torch.Tensor, name: str) -> None:
        """Log comprehensive tensor statistics"""
        if not tensor.numel():
            logger.debug(f"{name}: Empty tensor")
            return
        
        stats = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'mean': tensor.float().mean().item(),
            'std': tensor.float().std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'has_nan': torch.isnan(tensor).any().item(),
            'has_inf': torch.isinf(tensor).any().item()
        }
        
        logger.debug(f"{name} stats: {stats}")
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, name: str, check_finite: bool = True) -> bool:
        """Validate tensor for common issues"""
        if not isinstance(tensor, torch.Tensor):
            logger.error(f"{name}: Not a tensor, got {type(tensor)}")
            return False
        
        if tensor.numel() == 0:
            logger.warning(f"{name}: Empty tensor")
            return False
        
        if check_finite:
            if torch.isnan(tensor).any():
                logger.error(f"{name}: Contains NaN values")
                return False
            
            if torch.isinf(tensor).any():
                logger.error(f"{name}: Contains infinite values")
                return False
        
        return True
    
    @staticmethod
    def trace_function_calls(func):
        """Decorator to trace function calls"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args: {len(args)}, kwargs: {list(kwargs.keys())}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {duration:.4f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.4f}s: {e}")
                raise
        
        return wrapper

# =============================================================================
# SYSTEM UTILITIES
# =============================================================================

class SystemUtils:
    """System-level utilities"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information"""
        cpu_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
        
        memory_info = psutil.virtual_memory()._asdict()
        
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'devices': [
                    {
                        'name': torch.cuda.get_device_name(i),
                        'memory_total': torch.cuda.get_device_properties(i).total_memory,
                        'memory_allocated': torch.cuda.memory_allocated(i),
                        'memory_cached': torch.cuda.memory_reserved(i)
                    }
                    for i in range(torch.cuda.device_count())
                ]
            }
        
        return {
            'cpu': cpu_info,
            'memory': memory_info,
            'gpu': gpu_info,
            'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            'torch_version': torch.__version__,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """Ensure directory exists"""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def safe_file_write(content: str, filepath: str, backup: bool = True) -> bool:
        """Safely write content to file with backup"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create backup if file exists
            if backup and os.path.exists(filepath):
                backup_path = f"{filepath}.backup"
                import shutil
                shutil.copy2(filepath, backup_path)
            
            # Write content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        except Exception as e:
            logger.error(f"Failed to write file {filepath}: {e}")
            return False

# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def format_model_size(num_params: int) -> str:
    """Format model size in human-readable format"""
    for unit in ['', 'K', 'M', 'B', 'T']:
        if num_params < 1000:
            return f"{num_params:.1f}{unit}"
        num_params /= 1000
    return f"{num_params:.1f}P"

def format_memory_size(bytes_size: int) -> str:
    """Format memory size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}PB"

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Initialize logging configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def setup_warnings() -> None:
    """Setup warning filters"""
    # Filter out common warnings that don't affect functionality
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Initialize on import
setup_warnings()

# =============================================================================
# MAIN UTILITIES EXPORT
# =============================================================================

__all__ = [
    # Performance monitoring
    'PerformanceMonitor', 'perf_monitor', 'monitor_performance',
    
    # Memory management
    'MemoryTracker', 'memory_efficient',
    
    # Tensor utilities
    'TensorUtils',
    
    # Routing utilities
    'RoutingUtils',
    
    # Text processing
    'TextUtils',
    
    # Configuration
    'ConfigUtils',
    
    # Caching
    'CacheManager', 'cache_manager', 'cached',
    
    # Debugging
    'DebugUtils',
    
    # System utilities
    'SystemUtils',
    
    # Formatting utilities
    'format_model_size', 'format_memory_size', 'format_duration',
    
    # Initialization
    'initialize_logging', 'setup_warnings'
]
