"""
Metrics Collection and Monitoring System for Mamba Swarm
Tracks performance, resource usage, and model behavior
"""

import time
import threading
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import torch
import psutil
import numpy as np
from datetime import datetime, timedelta

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricPoint:
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class HistogramBucket:
    upper_bound: float
    count: int = 0

class Metric:
    """Base metric class"""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.lock = threading.Lock()
        self.created_at = time.time()

class Counter(Metric):
    """Counter metric - monotonically increasing"""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        super().__init__(name, description, labels)
        self.values = defaultdict(float)
    
    def inc(self, value: float = 1.0, **label_values):
        """Increment counter"""
        label_key = self._make_label_key(label_values)
        with self.lock:
            self.values[label_key] += value
    
    def get(self, **label_values) -> float:
        """Get counter value"""
        label_key = self._make_label_key(label_values)
        return self.values.get(label_key, 0.0)
    
    def _make_label_key(self, label_values: Dict[str, str]) -> str:
        """Create key from label values"""
        return "|".join(f"{k}={v}" for k, v in sorted(label_values.items()))

class Gauge(Metric):
    """Gauge metric - can go up and down"""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        super().__init__(name, description, labels)
        self.values = defaultdict(float)
    
    def set(self, value: float, **label_values):
        """Set gauge value"""
        label_key = self._make_label_key(label_values)
        with self.lock:
            self.values[label_key] = value
    
    def inc(self, value: float = 1.0, **label_values):
        """Increment gauge"""
        label_key = self._make_label_key(label_values)
        with self.lock:
            self.values[label_key] += value
    
    def dec(self, value: float = 1.0, **label_values):
        """Decrement gauge"""
        self.inc(-value, **label_values)
    
    def get(self, **label_values) -> float:
        """Get gauge value"""
        label_key = self._make_label_key(label_values)
        return self.values.get(label_key, 0.0)
    
    def _make_label_key(self, label_values: Dict[str, str]) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(label_values.items()))

class Histogram(Metric):
    """Histogram metric - tracks distribution of values"""
    
    def __init__(self, name: str, description: str, buckets: Optional[List[float]] = None, labels: Optional[List[str]] = None):
        super().__init__(name, description, labels)
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
        self.bucket_counts = defaultdict(lambda: defaultdict(int))
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
    
    def observe(self, value: float, **label_values):
        """Observe a value"""
        label_key = self._make_label_key(label_values)
        with self.lock:
            self.sums[label_key] += value
            self.counts[label_key] += 1
            
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[label_key][bucket] += 1
    
    def get_buckets(self, **label_values) -> Dict[float, int]:
        """Get bucket counts"""
        label_key = self._make_label_key(label_values)
        return dict(self.bucket_counts[label_key])
    
    def get_sum(self, **label_values) -> float:
        """Get sum of observed values"""
        label_key = self._make_label_key(label_values)
        return self.sums[label_key]
    
    def get_count(self, **label_values) -> int:
        """Get count of observations"""
        label_key = self._make_label_key(label_values)
        return self.counts[label_key]
    
    def _make_label_key(self, label_values: Dict[str, str]) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(label_values.items()))

class Summary(Metric):
    """Summary metric - tracks quantiles"""
    
    def __init__(self, name: str, description: str, quantiles: Optional[List[float]] = None, labels: Optional[List[str]] = None, max_age: float = 300.0):
        super().__init__(name, description, labels)
        self.quantiles = quantiles or [0.5, 0.9, 0.95, 0.99]
        self.max_age = max_age
        self.observations = defaultdict(lambda: deque())
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
    
    def observe(self, value: float, **label_values):
        """Observe a value"""
        label_key = self._make_label_key(label_values)
        timestamp = time.time()
        
        with self.lock:
            self.observations[label_key].append((timestamp, value))
            self.sums[label_key] += value
            self.counts[label_key] += 1
            
            # Clean old observations
            self._clean_old_observations(label_key, timestamp)
    
    def get_quantile(self, quantile: float, **label_values) -> float:
        """Get quantile value"""
        label_key = self._make_label_key(label_values)
        with self.lock:
            obs = self.observations[label_key]
            if not obs:
                return 0.0
            
            values = [v for _, v in obs]
            values.sort()
            index = int(quantile * len(values))
            return values[min(index, len(values) - 1)]
    
    def get_sum(self, **label_values) -> float:
        """Get sum of observed values"""
        label_key = self._make_label_key(label_values)
        return self.sums[label_key]
    
    def get_count(self, **label_values) -> int:
        """Get count of observations"""
        label_key = self._make_label_key(label_values)
        return self.counts[label_key]
    
    def _clean_old_observations(self, label_key: str, current_time: float):
        """Remove old observations"""
        cutoff_time = current_time - self.max_age
        obs = self.observations[label_key]
        
        while obs and obs[0][0] < cutoff_time:
            _, value = obs.popleft()
            self.sums[label_key] -= value
            self.counts[label_key] -= 1
    
    def _make_label_key(self, label_values: Dict[str, str]) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(label_values.items()))

class MetricsRegistry:
    """Registry for all metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.lock = threading.Lock()
    
    def register(self, metric: Metric):
        """Register a metric"""
        with self.lock:
            if metric.name in self.metrics:
                raise ValueError(f"Metric {metric.name} already registered")
            self.metrics[metric.name] = metric
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get metric by name"""
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all metrics"""
        return self.metrics.copy()

class MambaSwarmMetrics:
    """Metrics collector for Mamba Swarm"""
    
    def __init__(self):
        self.registry = MetricsRegistry()
        self.logger = logging.getLogger(__name__)
        self._setup_default_metrics()
        
        # System monitoring
        self.monitoring_thread = None
        self.monitoring_interval = 10.0  # seconds
        self.should_monitor = False
    
    def _setup_default_metrics(self):
        """Setup default metrics"""
        # Request metrics
        self.requests_total = Counter("requests_total", "Total number of requests", ["method", "endpoint", "status"])
        self.request_duration = Histogram("request_duration_seconds", "Request duration in seconds", labels=["method", "endpoint"])
        
        # Model metrics
        self.inference_duration = Histogram("inference_duration_seconds", "Inference duration in seconds", labels=["model_unit"])
        self.tokens_generated = Counter("tokens_generated_total", "Total tokens generated", ["model_unit"])
        self.model_load = Gauge("model_load", "Current model load", ["model_unit"])
        
        # System metrics
        self.memory_usage = Gauge("memory_usage_bytes", "Memory usage in bytes", ["type"])
        self.gpu_utilization = Gauge("gpu_utilization_percent", "GPU utilization percentage", ["device"])
        self.active_connections = Gauge("active_connections", "Number of active connections")
        
        # Swarm metrics
        self.encoder_utilization = Gauge("encoder_utilization", "Encoder utilization", ["encoder_id"])
        self.routing_decisions = Counter("routing_decisions_total", "Routing decisions", ["strategy", "target"])
        self.load_balancing_decisions = Counter("load_balancing_decisions_total", "Load balancing decisions", ["algorithm"])
        
        # Error metrics
        self.errors_total = Counter("errors_total", "Total number of errors", ["type", "component"])
        
        # Register all metrics
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Metric):
                self.registry.register(attr)
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring_thread is not None:
            return
        
        self.should_monitor = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started metrics monitoring")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.should_monitor = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            self.monitoring_thread = None
        self.logger.info("Stopped metrics monitoring")
    
    def _monitoring_loop(self):
        """System monitoring loop"""
        while self.should_monitor:
            try:
                self._collect_system_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        # Memory metrics
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used, type="system")
        self.memory_usage.set(memory.available, type="available")
        
        # GPU metrics
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                # GPU memory
                gpu_memory = torch.cuda.memory_allocated(i)
                self.memory_usage.set(gpu_memory, type=f"gpu_{i}")
                
                # GPU utilization (simplified)
                # In practice, you might use nvidia-ml-py for more detailed metrics
                utilization = min(100.0, (gpu_memory / torch.cuda.max_memory_allocated(i)) * 100) if torch.cuda.max_memory_allocated(i) > 0 else 0.0
                self.gpu_utilization.set(utilization, device=f"cuda:{i}")
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics"""
        self.requests_total.inc(method=method, endpoint=endpoint, status=str(status_code))
        self.request_duration.observe(duration, method=method, endpoint=endpoint)
    
    def record_inference(self, model_unit: str, duration: float, tokens: int):
        """Record inference metrics"""
        self.inference_duration.observe(duration, model_unit=model_unit)
        self.tokens_generated.inc(tokens, model_unit=model_unit)
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics"""
        self.errors_total.inc(type=error_type, component=component)
    
    def update_model_load(self, model_unit: str, load: float):
        """Update model load"""
        self.model_load.set(load, model_unit=model_unit)
    
    def update_encoder_utilization(self, encoder_id: str, utilization: float):
        """Update encoder utilization"""
        self.encoder_utilization.set(utilization, encoder_id=encoder_id)
    
    def record_routing_decision(self, strategy: str, target: str):
        """Record routing decision"""
        self.routing_decisions.inc(strategy=strategy, target=target)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {}
        
        for name, metric in self.registry.get_all_metrics().items():
            if isinstance(metric, Counter):
                summary[name] = {
                    "type": "counter",
                    "values": dict(metric.values)
                }
            elif isinstance(metric, Gauge):
                summary[name] = {
                    "type": "gauge",
                    "values": dict(metric.values)
                }
            elif isinstance(metric, Histogram):
                summary[name] = {
                    "type": "histogram",
                    "buckets": {k: dict(v) for k, v in metric.bucket_counts.items()},
                    "sums": dict(metric.sums),
                    "counts": dict(metric.counts)
                }
            elif isinstance(metric, Summary):
                summary[name] = {
                    "type": "summary",
                    "sums": dict(metric.sums),
                    "counts": dict(metric.counts),
                    "quantiles": {
                        q: {k: metric.get_quantile(q, **self._parse_label_key(k)) for k in metric.observations.keys()}
                        for q in metric.quantiles
                    }
                }
        
        return summary
    
    def _parse_label_key(self, label_key: str) -> Dict[str, str]:
        """Parse label key back to dictionary"""
        if not label_key:
            return {}
        
        labels = {}
        for pair in label_key.split("|"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        return labels
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        output = []
        
        for name, metric in self.registry.get_all_metrics().items():
            # Help text
            output.append(f"# HELP {name} {metric.description}")
            
            if isinstance(metric, Counter):
                output.append(f"# TYPE {name} counter")
                for label_key, value in metric.values.items():
                    labels = self._format_prometheus_labels(label_key)
                    output.append(f"{name}{labels} {value}")
            
            elif isinstance(metric, Gauge):
                output.append(f"# TYPE {name} gauge")
                for label_key, value in metric.values.items():
                    labels = self._format_prometheus_labels(label_key)
                    output.append(f"{name}{labels} {value}")
            
            elif isinstance(metric, Histogram):
                output.append(f"# TYPE {name} histogram")
                for label_key in metric.bucket_counts.keys():
                    labels_dict = self._parse_label_key(label_key)
                    
                    # Buckets
                    for bucket, count in metric.bucket_counts[label_key].items():
                        bucket_labels = {**labels_dict, "le": str(bucket)}
                        bucket_label_str = self._format_prometheus_labels_dict(bucket_labels)
                        output.append(f"{name}_bucket{bucket_label_str} {count}")
                    
                    # Sum and count
                    base_labels = self._format_prometheus_labels(label_key)
                    output.append(f"{name}_sum{base_labels} {metric.sums[label_key]}")
                    output.append(f"{name}_count{base_labels} {metric.counts[label_key]}")
            
            elif isinstance(metric, Summary):
                output.append(f"# TYPE {name} summary")
                for label_key in metric.observations.keys():
                    labels_dict = self._parse_label_key(label_key)
                    
                    # Quantiles
                    for quantile in metric.quantiles:
                        quantile_labels = {**labels_dict, "quantile": str(quantile)}
                        quantile_label_str = self._format_prometheus_labels_dict(quantile_labels)
                        quantile_value = metric.get_quantile(quantile, **labels_dict)
                        output.append(f"{name}{quantile_label_str} {quantile_value}")
                    
                    # Sum and count
                    base_labels = self._format_prometheus_labels(label_key)
                    output.append(f"{name}_sum{base_labels} {metric.sums[label_key]}")
                    output.append(f"{name}_count{base_labels} {metric.counts[label_key]}")
            
            output.append("")  # Empty line between metrics
        
        return "\n".join(output)
    
    def _format_prometheus_labels(self, label_key: str) -> str:
        """Format labels for Prometheus"""
        if not label_key:
            return ""
        
        labels = self._parse_label_key(label_key)
        return self._format_prometheus_labels_dict(labels)
    
    def _format_prometheus_labels_dict(self, labels: Dict[str, str]) -> str:
        """Format label dictionary for Prometheus"""
        if not labels:
            return ""
        
        formatted_labels = []
        for k, v in sorted(labels.items()):
            # Escape quotes and backslashes
            escaped_value = v.replace("\\", "\\\\").replace('"', '\\"')
            formatted_labels.append(f'{k}="{escaped_value}"')
        
        return "{" + ",".join(formatted_labels) + "}"

# Context managers for timing
class timer:
    """Context manager for timing operations"""
    
    def __init__(self, metric: Histogram, **labels):
        self.metric = metric
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metric.observe(duration, **self.labels)

class request_timer:
    """Context manager for timing requests"""
    
    def __init__(self, metrics: MambaSwarmMetrics, method: str, endpoint: str):
        self.metrics = metrics
        self.method = method
        self.endpoint = endpoint
        self.start_time = None
        self.status_code = 200
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.status_code = 500
        
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics.record_request(self.method, self.endpoint, self.status_code, duration)
    
    def set_status(self, status_code: int):
        """Set the response status code"""
        self.status_code = status_code

# Decorator for automatic metrics collection
def measure_time(metric_name: str, **labels):
    """Decorator to measure function execution time"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Assume first argument is self and has metrics attribute
            if args and hasattr(args[0], 'metrics'):
                metrics = args[0].metrics
                metric = metrics.registry.get_metric(metric_name)
                if metric and isinstance(metric, Histogram):
                    with timer(metric, **labels):
                        return func(*args, **kwargs)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Metrics aggregator for multiple instances
class MetricsAggregator:
    """Aggregates metrics from multiple Mamba Swarm instances"""
    
    def __init__(self):
        self.instances: Dict[str, MambaSwarmMetrics] = {}
        self.lock = threading.Lock()
    
    def add_instance(self, instance_id: str, metrics: MambaSwarmMetrics):
        """Add metrics instance"""
        with self.lock:
            self.instances[instance_id] = metrics
    
    def remove_instance(self, instance_id: str):
        """Remove metrics instance"""
        with self.lock:
            self.instances.pop(instance_id, None)
    
    def get_aggregated_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary"""
        aggregated = defaultdict(lambda: defaultdict(float))
        
        with self.lock:
            for instance_id, metrics in self.instances.items():
                summary = metrics.get_metrics_summary()
                
                for metric_name, metric_data in summary.items():
                    if metric_data["type"] in ["counter", "gauge"]:
                        for label_key, value in metric_data["values"].items():
                            key = f"{metric_name}_{label_key}" if label_key else metric_name
                            
                            if metric_data["type"] == "counter":
                                aggregated[key]["sum"] += value
                            else:  # gauge
                                aggregated[key]["avg"] = (aggregated[key].get("avg", 0) + value) / 2
                                aggregated[key]["instances"] = aggregated[key].get("instances", 0) + 1
        
        return dict(aggregated)

# FastAPI integration
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse

def add_metrics_endpoints(app: FastAPI, metrics: MambaSwarmMetrics):
    """Add metrics endpoints to FastAPI app"""
    
    @app.get("/metrics")
    async def get_metrics():
        """Get metrics in JSON format"""
        return metrics.get_metrics_summary()
    
    @app.get("/metrics/prometheus")
    async def get_prometheus_metrics():
        """Get metrics in Prometheus format"""
        prometheus_data = metrics.export_prometheus_format()
        return PlainTextResponse(prometheus_data, media_type="text/plain")
    
    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        """Middleware to collect request metrics"""
        method = request.method
        path = request.url.path
        
        with request_timer(metrics, method, path) as timer_ctx:
            response = await call_next(request)
            timer_ctx.set_status(response.status_code)
            return response

# Example usage
if __name__ == "__main__":
    # Create metrics instance
    metrics = MambaSwarmMetrics()
    metrics.start_monitoring()
    
    # Example metric recording
    metrics.record_request("POST", "/generate", 200, 0.5)
    metrics.record_inference("encoder_1", 0.3, 50)
    metrics.update_encoder_utilization("encoder_1", 0.8)
    
    # Get summary
    summary = metrics.get_metrics_summary()
    print(json.dumps(summary, indent=2))
    
    # Export Prometheus format
    prometheus_data = metrics.export_prometheus_format()
    print("\nPrometheus format:")
    print(prometheus_data)
    
    metrics.stop_monitoring() 