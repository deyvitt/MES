"""
Main entry point for Mamba Swarm
100 units of 70M parameter Mamba encoders for distributed language modeling
"""

import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import core components
from core.config import MambaSwarmConfig
from system.mambaSwarm import SwarmEngine
from system.inference import InferenceEngine
from api.api_server import run_server
from api.load_balancer import run_load_balancer, LoadBalancingStrategy
from training.trainer import DistributedTrainer
from monitoring.metrics import MambaSwarmMetrics
from monitoring.profiler import MambaSwarmProfiler
from monitoring.evaluator import MambaSwarmEvaluator
from checkpoints.checkpoint_manager import CheckpointManager
from training.trainer import setup_logging, get_device_info

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description="Mamba Swarm - Distributed Language Model")
    
    # Main mode selection
    parser.add_argument("mode", choices=["train", "serve", "evaluate", "load_balance"], 
                       help="Operation mode")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config/default.yaml",
                       help="Configuration file path")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint to load")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--data-path", type=str, default="data/",
                       help="Training data path")
    
    # Serving arguments
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes")
    
    # Load balancer arguments
    parser.add_argument("--servers", type=str, nargs="+",
                       help="Backend server addresses (host:port)")
    parser.add_argument("--strategy", type=str, default="resource_aware",
                       choices=["round_robin", "least_connections", "weighted_round_robin", 
                               "least_response_time", "hash_based", "resource_aware"],
                       help="Load balancing strategy")
    
    # Evaluation arguments
    parser.add_argument("--eval-data", type=str, default="data/eval/",
                       help="Evaluation data path")
    parser.add_argument("--output-report", type=str, default=None,
                       help="Evaluation report output path")
    
    # System arguments
    parser.add_argument("--num-encoders", type=int, default=100,
                       help="Number of Mamba encoders")
    parser.add_argument("--encoder-params", type=int, default=70000000,
                       help="Parameters per encoder (70M)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda, cpu, auto)")
    parser.add_argument("--distributed", action="store_true",
                       help="Enable distributed training")
    
    # Monitoring arguments
    parser.add_argument("--enable-metrics", action="store_true",
                       help="Enable metrics collection")
    parser.add_argument("--enable-profiling", action="store_true",
                       help="Enable performance profiling")
    parser.add_argument("--metrics-port", type=int, default=9090,
                       help="Metrics server port")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Log file path")
    
    return parser

async def train_mode(args, config: MambaSwarmConfig):
    """Training mode"""
    logging.info("Starting Mamba Swarm training...")
    
    # Initialize components
    metrics = MambaSwarmMetrics() if args.enable_metrics else None
    profiler = MambaSwarmProfiler() if args.enable_profiling else None
    
    # Initialize swarm engine
    swarm_engine = SwarmEngine(config)
    swarm_engine.initialize()
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        max_checkpoints=config.max_checkpoints,
        save_interval=config.save_interval
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint_data = checkpoint_manager.load_checkpoint(args.checkpoint)
        if checkpoint_data:
            swarm_engine.load_state_dict(checkpoint_data["model_state"])
            logging.info(f"Loaded checkpoint: {args.checkpoint}")
    
    # Initialize trainer
    trainer = DistributedTrainer(
        swarm_engine=swarm_engine,
        config=config,
        checkpoint_manager=checkpoint_manager,
        metrics=metrics,
        profiler=profiler
    )
    
    try:
        # Start monitoring
        if metrics:
            metrics.start_monitoring()
        if profiler:
            profiler.start_profiling()
        
        # Train model
        await trainer.train(
            data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
    finally:
        # Cleanup
        if metrics:
            metrics.stop_monitoring()
        if profiler:
            profiler.cleanup()
        swarm_engine.shutdown()

def serve_mode(args, config: MambaSwarmConfig):
    """API serving mode"""
    logging.info("Starting Mamba Swarm API server...")
    
    # Run API server
    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers
    )

def load_balance_mode(args, config: MambaSwarmConfig):
    """Load balancer mode"""
    logging.info("Starting Mamba Swarm load balancer...")
    
    # Parse server addresses
    servers = []
    for server_addr in args.servers or []:
        if ":" in server_addr:
            host, port = server_addr.split(":", 1)
            servers.append((host, int(port)))
        else:
            servers.append((server_addr, 8000))  # Default port
    
    if not servers:
        logging.error("No backend servers specified")
        return
    
    # Map strategy name to enum
    strategy_map = {
        "round_robin": LoadBalancingStrategy.ROUND_ROBIN,
        "least_connections": LoadBalancingStrategy.LEAST_CONNECTIONS,
        "weighted_round_robin": LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
        "least_response_time": LoadBalancingStrategy.LEAST_RESPONSE_TIME,
        "hash_based": LoadBalancingStrategy.HASH_BASED,
        "resource_aware": LoadBalancingStrategy.RESOURCE_AWARE
    }
    
    strategy = strategy_map.get(args.strategy, LoadBalancingStrategy.RESOURCE_AWARE)
    
    # Run load balancer
    run_load_balancer(
        servers=servers,
        host=args.host,
        port=args.port,
        strategy=strategy
    )

async def evaluate_mode(args, config: MambaSwarmConfig):
    """Evaluation mode"""
    logging.info("Starting Mamba Swarm evaluation...")
    
    # Initialize swarm engine
    swarm_engine = SwarmEngine(config)
    swarm_engine.initialize()
    
    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        checkpoint_data = checkpoint_manager.load_checkpoint(args.checkpoint)
        if checkpoint_data:
            swarm_engine.load_state_dict(checkpoint_data["model_state"])
            logging.info(f"Loaded checkpoint: {args.checkpoint}")
    
    # Initialize evaluator
    evaluator = MambaSwarmEvaluator(swarm_engine, config.__dict__)
    
    try:
        # Run comprehensive evaluation
        result = evaluator.run_comprehensive_evaluation()
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"Total Metrics: {len(result.individual_metrics)}")
        
        # Print top metrics
        print(f"\nTop Metrics:")
        for metric in result.individual_metrics[:10]:
            print(f"  {metric.metric_name}: {metric.score:.3f}")
        
        # Export report
        output_path = args.output_report or f"evaluation_report_{int(result.timestamp)}.json"
        report_file = evaluator.export_evaluation_report(result, output_path)
        print(f"\nDetailed report saved to: {report_file}")
        
    finally:
        swarm_engine.shutdown()

def validate_config(args) -> MambaSwarmConfig:
    """Validate and create configuration"""
    
    # Load base configuration
    if os.path.exists(args.config):
        config = MambaSwarmConfig.from_file(args.config)
    else:
        logging.warning(f"Config file {args.config} not found, using defaults")
        config = MambaSwarmConfig()
    
    # Override with command line arguments
    if args.num_encoders:
        config.num_encoders = args.num_encoders
    if args.encoder_params:
        config.encoder_params = args.encoder_params
    
    # Device configuration
    if args.device == "auto":
        device_info = get_device_info()
        config.device = "cuda" if device_info["cuda_available"] else "cpu"
    else:
        config.device = args.device
    
    # Validate configuration
    total_params = config.num_encoders * config.encoder_params
    logging.info(f"Configuration: {config.num_encoders} encoders × {config.encoder_params/1e6:.0f}M params = {total_params/1e9:.1f}B total parameters")
    
    return config

def main():
    """Main entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        level=getattr(logging, args.log_level),
        log_file=args.log_file
    )
    
    # Print banner
    print("=" * 60)
    print("🐍 Mamba Swarm - Distributed Language Model")
    print("100 × 70M Parameter Mamba Encoders")
    print("=" * 60)
    
    # Validate configuration
    try:
        config = validate_config(args)
    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Print system information
    device_info = get_device_info()
    logging.info(f"System: {device_info['cpu_count']} CPUs, {device_info['memory_gb']:.1f}GB RAM")
    if device_info["cuda_available"]:
        logging.info(f"GPU: {device_info['gpu_count']} devices, {device_info['gpu_memory_gb']:.1f}GB VRAM")
    
    # Run mode-specific logic
    try:
        if args.mode == "train":
            asyncio.run(train_mode(args, config))
        elif args.mode == "serve":
            serve_mode(args, config)
        elif args.mode == "load_balance":
            load_balance_mode(args, config)
        elif args.mode == "evaluate":
            asyncio.run(evaluate_mode(args, config))
        else:
            logging.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logging.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)
    
    logging.info("Mamba Swarm shutdown complete")

def print_usage_examples():
    """Print usage examples"""
    examples = """
Usage Examples:

1. Training:
   python main.py train --data-path ./data/train --epochs 10 --batch-size 8 --enable-metrics

2. Serving:
   python main.py serve --host 0.0.0.0 --port 8000 --checkpoint best_model.pt

3. Load Balancing:
   python main.py load_balance --servers localhost:8000 localhost:8001 localhost:8002 --strategy resource_aware

4. Evaluation:
   python main.py evaluate --checkpoint best_model.pt --eval-data ./data/eval --output-report eval_results.json

5. Distributed Training:
   python main.py train --distributed --num-encoders 100 --batch-size 4 --enable-profiling

Configuration File Example (config.yaml):
---
num_encoders: 100
encoder_params: 70000000
hidden_size: 2048
num_layers: 32
vocab_size: 50000
max_sequence_length: 2048
device: "auto"
checkpoint_dir: "./checkpoints"
max_checkpoints: 10
save_interval: 1000
learning_rate: 1e-4
warmup_steps: 1000
weight_decay: 0.01
gradient_clip_norm: 1.0
mixed_precision: true
gradient_accumulation_steps: 8
"""
    print(examples)

if __name__ == "__main__":
    # Check for help with examples
    if len(sys.argv) == 2 and sys.argv[1] in ["--help-examples", "-he"]:
        print_usage_examples()
        sys.exit(0)
    
    main() 