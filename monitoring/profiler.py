"""
Performance Profiler for Mamba Swarm
Advanced profiling tools for performance analysis and optimization
"""

import time
import cProfile
import pstats
import io
import threading
import functools
import traceback
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import logging
import json
from datetime import datetime
import os
import gc

@dataclass
class ProfileResult:
    function_name: str
    total_time: float
    cumulative_time: float
    call_count: int
    per_call_time: float
    filename: str
    line_number: int

@dataclass
class MemorySnapshot:
    timestamp: float
    total_memory: float
    gpu_memory: float
    python_objects: int
    tensor_count: int
    cache_size: float

@dataclass
class PerformanceProfile:
    timestamp: float
    duration: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    function_calls: List[ProfileResult]
    memory_snapshots: List[MemorySnapshot]
    bottlenecks: List[str]
    recommendations: List[str]

class FunctionTimer:
    """Timer for individual function calls"""
    
    def __init__(self, name: str):
        self.name = name
        self.calls = []
        self.total_time = 0.0
        self.call_count = 0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.lock = threading.Lock()
    
    def add_call(self, duration: float):
        """Add a function call duration"""
        with self.lock:
            self.calls.append(duration)
            self.total_time += duration
            self.call_count += 1
            self.min_time = min(self.min_time, duration)
            self.max_time = max(self.max_time, duration)
            
            # Keep only recent calls
            if len(self.calls) > 1000:
                old_call = self.calls.pop(0)
                self.total_time -= old_call
                self.call_count -= 1
    
    @property
    def avg_time(self) -> float:
        return self.total_time / max(self.call_count, 1)
    
    @property
    def percentile_95(self) -> float:
        if not self.calls:
            return 0.0
        sorted_calls = sorted(self.calls)
        index = int(0.95 * len(sorted_calls))
        return sorted_calls[min(index, len(sorted_calls) - 1)]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_time": self.total_time,
            "call_count": self.call_count,
            "avg_time": self.avg_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0.0,
            "max_time": self.max_time,
            "percentile_95": self.percentile_95
        }

class MemoryProfiler:
    """Memory usage profiler"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.snapshots = deque(maxlen=1000)
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start memory monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Memory monitoring loop"""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                with self.lock:
                    self.snapshots.append(snapshot)
                time.sleep(self.sample_interval)
            except Exception as e:
                logging.error(f"Memory monitoring error: {e}")
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot"""
        # System memory
        memory = psutil.virtual_memory()
        total_memory = memory.used / (1024**3)  # GB
        
        # GPU memory
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        
        # Python objects
        python_objects = len(gc.get_objects())
        
        # Tensor count
        tensor_count = 0
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor):
                tensor_count += 1
        
        # Cache size estimation
        cache_size = 0.0  # Could be calculated based on specific cache implementations
        
        return MemorySnapshot(
            timestamp=time.time(),
            total_memory=total_memory,
            gpu_memory=gpu_memory,
            python_objects=python_objects,
            tensor_count=tensor_count,
            cache_size=cache_size
        )
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage"""
        with self.lock:
            if not self.snapshots:
                return 0.0
            return max(snapshot.total_memory + snapshot.gpu_memory for snapshot in self.snapshots)
    
    def get_memory_trend(self) -> List[float]:
        """Get memory usage trend"""
        with self.lock:
            return [snapshot.total_memory + snapshot.gpu_memory for snapshot in self.snapshots]

class CPUProfiler:
    """CPU profiling using cProfile"""
    
    def __init__(self):
        self.profiler = None
        self.profiling = False
        self.lock = threading.Lock()
    
    def start_profiling(self):
        """Start CPU profiling"""
        with self.lock:
            if self.profiling:
                return
            
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.profiling = True
    
    def stop_profiling(self) -> List[ProfileResult]:
        """Stop CPU profiling and return results"""
        with self.lock:
            if not self.profiling or not self.profiler:
                return []
            
            self.profiler.disable()
            self.profiling = False
            
            # Analyze results
            s = io.StringIO()
            stats = pstats.Stats(self.profiler, stream=s)
            stats.sort_stats('cumulative')
            
            results = []
            for func, (call_count, total_time, cumulative_time, callers) in stats.stats.items():
                filename, line_number, function_name = func
                
                result = ProfileResult(
                    function_name=function_name,
                    total_time=total_time,
                    cumulative_time=cumulative_time,
                    call_count=call_count,
                    per_call_time=total_time / call_count if call_count > 0 else 0.0,
                    filename=filename,
                    line_number=line_number
                )
                results.append(result)
            
            # Sort by cumulative time
            results.sort(key=lambda x: x.cumulative_time, reverse=True)
            return results

class GPUProfiler:
    """GPU profiling for CUDA operations"""
    
    def __init__(self):
        self.events = []
        self.profiling = False
        self.lock = threading.Lock()
    
    def start_profiling(self):
        """Start GPU profiling"""
        if not torch.cuda.is_available():
            return
        
        with self.lock:
            if self.profiling:
                return
            
            self.events = []
            self.profiling = True
            torch.cuda.synchronize()
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop GPU profiling"""
        if not torch.cuda.is_available():
            return {}
        
        with self.lock:
            if not self.profiling:
                return {}
            
            torch.cuda.synchronize()
            self.profiling = False
            
            # Calculate GPU metrics
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            cached_memory = torch.cuda.memory_reserved() / (1024**3)
            
            return {
                "total_memory_gb": total_memory,
                "allocated_memory_gb": allocated_memory,
                "cached_memory_gb": cached_memory,
                "memory_utilization": allocated_memory / total_memory * 100,
                "events": len(self.events)
            }
    
    @contextmanager
    def profile_operation(self, name: str):
        """Context manager for profiling GPU operations"""
        if not torch.cuda.is_available() or not self.profiling:
            yield
            return
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        try:
            yield
        finally:
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event)
            with self.lock:
                self.events.append({
                    "name": name,
                    "duration_ms": elapsed_time,
                    "timestamp": time.time()
                })

class MambaSwarmProfiler:
    """Comprehensive profiler for Mamba Swarm"""
    
    def __init__(self, enable_memory_monitoring: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Initialize profilers
        self.cpu_profiler = CPUProfiler()
        self.memory_profiler = MemoryProfiler()
        self.gpu_profiler = GPUProfiler()
        
        # Function timers
        self.function_timers: Dict[str, FunctionTimer] = {}
        self.timer_lock = threading.Lock()
        
        # Profiling state
        self.profiling_active = False
        self.profile_start_time = 0.0
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        
        # Start memory monitoring if enabled
        if enable_memory_monitoring:
            self.memory_profiler.start_monitoring()
    
    def start_profiling(self, include_cpu: bool = True, include_gpu: bool = True):
        """Start comprehensive profiling"""
        if self.profiling_active:
            self.logger.warning("Profiling already active")
            return
        
        self.profile_start_time = time.time()
        self.profiling_active = True
        
        if include_cpu:
            self.cpu_profiler.start_profiling()
        
        if include_gpu:
            self.gpu_profiler.start_profiling()
        
        self.logger.info("Started performance profiling")
    
    def stop_profiling(self) -> PerformanceProfile:
        """Stop profiling and return results"""
        if not self.profiling_active:
            self.logger.warning("Profiling not active")
            return None
        
        end_time = time.time()
        duration = end_time - self.profile_start_time
        self.profiling_active = False
        
        # Get CPU profile
        cpu_results = self.cpu_profiler.stop_profiling()
        
        # Get GPU profile
        gpu_results = self.gpu_profiler.stop_profiling()
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        
        gpu_usage = 0.0
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        
        # Get memory snapshots
        memory_snapshots = list(self.memory_profiler.snapshots)
        
        # Analyze bottlenecks
        bottlenecks = self._analyze_bottlenecks(cpu_results, gpu_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(cpu_results, gpu_results, memory_snapshots)
        
        profile = PerformanceProfile(
            timestamp=end_time,
            duration=duration,
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            gpu_usage=gpu_usage,
            function_calls=cpu_results,
            memory_snapshots=memory_snapshots,
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )
        
        self.performance_history.append(profile)
        self.logger.info(f"Completed performance profiling (duration: {duration:.2f}s)")
        
        return profile
    
    def _analyze_bottlenecks(self, cpu_results: List[ProfileResult], gpu_results: Dict[str, Any]) -> List[str]:
        """Analyze performance bottlenecks"""
        bottlenecks = []
        
        # CPU bottlenecks
        if cpu_results:
            top_cpu_functions = cpu_results[:5]
            for func in top_cpu_functions:
                if func.cumulative_time > 1.0:  # More than 1 second
                    bottlenecks.append(f"CPU: {func.function_name} ({func.cumulative_time:.2f}s)")
        
        # Memory bottlenecks
        peak_memory = self.memory_profiler.get_peak_memory()
        if peak_memory > 8.0:  # More than 8GB
            bottlenecks.append(f"Memory: High usage ({peak_memory:.2f}GB)")
        
        # GPU bottlenecks
        if gpu_results and gpu_results.get("memory_utilization", 0) > 90:
            bottlenecks.append("GPU: High memory utilization")
        
        return bottlenecks
    
    def _generate_recommendations(self, cpu_results: List[ProfileResult], 
                                gpu_results: Dict[str, Any], 
                                memory_snapshots: List[MemorySnapshot]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # CPU recommendations
        if cpu_results:
            slow_functions = [f for f in cpu_results if f.per_call_time > 0.1]
            if slow_functions:
                recommendations.append("Consider optimizing slow functions or using caching")
        
        # Memory recommendations
        if memory_snapshots:
            tensor_counts = [s.tensor_count for s in memory_snapshots]
            if tensor_counts and max(tensor_counts) > 10000:
                recommendations.append("High tensor count detected - consider tensor cleanup")
        
        # GPU recommendations
        if gpu_results:
            if gpu_results.get("memory_utilization", 0) > 85:
                recommendations.append("Consider reducing batch size or using gradient checkpointing")
        
        return recommendations
    
    def profile_function(self, func_name: str):
        """Decorator for profiling individual functions"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    
                    with self.timer_lock:
                        if func_name not in self.function_timers:
                            self.function_timers[func_name] = FunctionTimer(func_name)
                        self.function_timers[func_name].add_call(duration)
            
            return wrapper
        return decorator
    
    @contextmanager
    def profile_block(self, block_name: str):
        """Context manager for profiling code blocks"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            with self.timer_lock:
                if block_name not in self.function_timers:
                    self.function_timers[block_name] = FunctionTimer(block_name)
                self.function_timers[block_name].add_call(duration)
    
    def get_function_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all profiled functions"""
        with self.timer_lock:
            return {name: timer.get_stats() for name, timer in self.function_timers.items()}
    
    def export_profile_report(self, filename: Optional[str] = None) -> str:
        """Export comprehensive profile report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mamba_swarm_profile_{timestamp}.json"
        
        report = {
            "timestamp": time.time(),
            "profiler_stats": {
                "function_timers": self.get_function_stats(),
                "peak_memory_gb": self.memory_profiler.get_peak_memory(),
                "memory_trend": self.memory_profiler.get_memory_trend()[-50:],  # Last 50 samples
            },
            "performance_history": [
                {
                    "timestamp": p.timestamp,
                    "duration": p.duration,
                    "cpu_usage": p.cpu_usage,
                    "memory_usage": p.memory_usage,
                    "gpu_usage": p.gpu_usage,
                    "bottlenecks": p.bottlenecks,
                    "recommendations": p.recommendations
                }
                for p in list(self.performance_history)[-10:]  # Last 10 profiles
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Profile report exported to {filename}")
        return filename
    
    def cleanup(self):
        """Cleanup profiler resources"""
        self.memory_profiler.stop_monitoring()
        if self.profiling_active:
            self.stop_profiling()

# Utility functions and decorators
def profile_inference(profiler: MambaSwarmProfiler):
    """Decorator for profiling inference functions"""
    return profiler.profile_function("inference")

def profile_training_step(profiler: MambaSwarmProfiler):
    """Decorator for profiling training steps"""
    return profiler.profile_function("training_step")

def profile_forward_pass(profiler: MambaSwarmProfiler):
    """Decorator for profiling forward passes"""
    return profiler.profile_function("forward_pass")

# Example usage
if __name__ == "__main__":
    # Create profiler
    profiler = MambaSwarmProfiler()
    
    # Start profiling
    profiler.start_profiling()
    
    # Simulate some work
    @profiler.profile_function("test_function")
    def test_function():
        time.sleep(0.1)
        return "result"
    
    # Run test
    for i in range(10):
        test_function()
    
    # Use context manager
    with profiler.profile_block("test_block"):
        time.sleep(0.05)
    
    # Stop profiling
    profile_result = profiler.stop_profiling()
    
    # Print results
    if profile_result:
        print(f"Profile duration: {profile_result.duration:.2f}s")
        print(f"CPU usage: {profile_result.cpu_usage:.1f}%")
        print(f"Memory usage: {profile_result.memory_usage:.1f}%")
        print(f"Bottlenecks: {profile_result.bottlenecks}")
        print(f"Recommendations: {profile_result.recommendations}")
    
    # Export report
    report_file = profiler.export_profile_report()
    print(f"Report saved to: {report_file}")
    
    # Cleanup
    profiler.cleanup() 