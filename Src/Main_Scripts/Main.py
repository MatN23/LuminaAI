# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import traceback
import psutil
import gc
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# DeepSpeed imports
try:
    import deepspeed
    import torch
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    logging.warning("DeepSpeed not available")

# Import our modules with fallbacks
try:
    from config.config_manager import Config, ConfigPresets
except ImportError:
    try:
        from config_manager import Config, ConfigPresets
    except ImportError:
        print("ERROR: Could not import config classes")
        sys.exit(1)

try:
    from training.orchestrator import AdaptiveTrainingOrchestrator, TrainingMetrics
except ImportError:
    try:
        from orchestrator import AdaptiveTrainingOrchestrator, TrainingMetrics
    except ImportError:
        AdaptiveTrainingOrchestrator = None
        TrainingMetrics = None
        print("Warning: Orchestrator not available, using basic training")

try:
    from utils.data_processing import process_oasst_data, validate_data_comprehensive
    from utils.environment import validate_environment, estimate_training_time
    from utils.reporting import create_data_summary_report
except ImportError:
    # Create fallback functions
    def process_oasst_data(*args, **kwargs):
        print("Data processing not available")
        return False
    
    def validate_data_comprehensive(*args, **kwargs):
        return {"status": "validation_not_available"}
    
    def validate_environment():
        return "environment_validation_not_available"
    
    def estimate_training_time(*args, **kwargs):
        return {"estimated_hours": 0}
    
    def create_data_summary_report(*args, **kwargs):
        return "report_not_available"

try:
    from core.tokenizer import ConversationTokenizer
    from core.model import estimate_parameters, DeepSeekTransformer, DeepSeekConfig
    from core.dataset import ConversationDataset, StreamingConversationDataset, create_memory_efficient_dataloader
except ImportError:
    try:
        from tokenizer import ConversationTokenizer
        from model import estimate_parameters, DeepSeekTransformer, DeepSeekConfig
        from dataset import ConversationDataset, StreamingConversationDataset, create_memory_efficient_dataloader
    except ImportError:
        print("ERROR: Could not import core modules")
        sys.exit(1)

try:
    from training.trainer import EnhancedConversationTrainer, DeepSpeedConfigGenerator, debug_dataloader
    from training.checkpoint import CheckpointManager
except ImportError:
    try:
        from trainer import EnhancedConversationTrainer, DeepSpeedConfigGenerator, debug_dataloader
        from checkpoint import CheckpointManager
    except ImportError:
        print("Warning: Enhanced trainer not available")
        EnhancedConversationTrainer = None
        DeepSpeedConfigGenerator = None
        debug_dataloader = None
        CheckpointManager = None


def create_argument_parser():
    """Create argument parser focused on training workflow and system features."""
    parser = argparse.ArgumentParser(
        description="Enhanced DeepSpeed MoE Training System with Advanced Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Advanced Training Features:
  
  # Resume training from latest checkpoint
  python Main.py --resume
  python Main.py --resume latest
  python Main.py --resume best
  python Main.py --resume checkpoints/specific_checkpoint.pt
  
  # Continue training with different experiment name
  python Main.py --resume --continue-as new_experiment_name
  
  # Analyze model without training
  python Main.py --analyze-model
  python Main.py --analyze-model --checkpoint best
  
  # Data validation and preprocessing
  python Main.py --validate-data
  python Main.py --process-data input.jsonl output.jsonl
  
  # System and environment checks
  python Main.py --check-environment
  python Main.py --benchmark-performance
  
  # Export model for inference
  python Main.py --export-model --checkpoint best --format pytorch
  
  # Training monitoring and debugging
  python Main.py --debug-setup
  python Main.py --profile-memory
  
  # Backup and restore
  python Main.py --create-backup
  python Main.py --list-checkpoints
        """
    )
    
    # === TRAINING WORKFLOW ===
    workflow_group = parser.add_argument_group('Training Workflow')
    workflow_group.add_argument(
        '--resume', 
        nargs='?',
        const='latest',
        metavar='CHECKPOINT',
        help='Resume training from checkpoint (latest/best/path). Default: latest'
    )
    
    workflow_group.add_argument(
        '--continue-as',
        type=str,
        metavar='NAME',
        help='Continue training with a new experiment name'
    )
    
    workflow_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Test configuration without training'
    )
    
    workflow_group.add_argument(
        '--force-restart',
        action='store_true',
        help='Ignore existing checkpoints and start fresh'
    )
    
    # === MODEL ANALYSIS ===
    analysis_group = parser.add_argument_group('Model Analysis')
    analysis_group.add_argument(
        '--analyze-model',
        action='store_true',
        help='Analyze model architecture and parameters'
    )
    
    analysis_group.add_argument(
        '--checkpoint', '--ckpt',
        type=str,
        metavar='PATH',
        help='Checkpoint path for analysis (latest/best/path)'
    )
    
    analysis_group.add_argument(
        '--model-stats',
        action='store_true',
        help='Show detailed model statistics'
    )
    
    analysis_group.add_argument(
        '--memory-analysis',
        action='store_true',
        help='Analyze memory requirements'
    )
    
    # === DATA MANAGEMENT ===
    data_group = parser.add_argument_group('Data Management')
    data_group.add_argument(
        '--validate-data',
        action='store_true',
        help='Validate training data quality'
    )
    
    data_group.add_argument(
        '--process-data',
        nargs=2,
        metavar=('INPUT', 'OUTPUT'),
        help='Process raw data: input_file output_file'
    )
    
    data_group.add_argument(
        '--data-report',
        action='store_true',
        help='Generate data summary report'
    )
    
    data_group.add_argument(
        '--create-shards',
        type=int,
        metavar='N',
        help='Split data into N shards for distributed training'
    )
    
    # === SYSTEM DIAGNOSTICS ===
    system_group = parser.add_argument_group('System Diagnostics')
    system_group.add_argument(
        '--check-environment',
        action='store_true',
        help='Validate training environment and dependencies'
    )
    
    system_group.add_argument(
        '--benchmark-performance',
        action='store_true',
        help='Benchmark training performance'
    )
    
    system_group.add_argument(
        '--profile-memory',
        action='store_true',
        help='Profile memory usage during training'
    )
    
    system_group.add_argument(
        '--debug-setup',
        action='store_true',
        help='Debug training setup and configuration'
    )
    
    system_group.add_argument(
        '--test-deepspeed',
        action='store_true',
        help='Test DeepSpeed configuration and setup'
    )
    
    # === MODEL EXPORT ===
    export_group = parser.add_argument_group('Model Export')
    export_group.add_argument(
        '--export-model',
        action='store_true',
        help='Export trained model for inference'
    )
    
    export_group.add_argument(
        '--format',
        choices=['pytorch', 'onnx', 'tensorrt', 'huggingface'],
        default='pytorch',
        help='Export format (default: pytorch)'
    )
    
    export_group.add_argument(
        '--output-dir',
        type=str,
        default='exported_models',
        help='Output directory for exported model'
    )
    
    # === CHECKPOINT MANAGEMENT ===
    checkpoint_group = parser.add_argument_group('Checkpoint Management')
    checkpoint_group.add_argument(
        '--list-checkpoints',
        action='store_true',
        help='List all available checkpoints'
    )
    
    checkpoint_group.add_argument(
        '--create-backup',
        action='store_true',
        help='Create backup of current training state'
    )
    
    checkpoint_group.add_argument(
        '--clean-checkpoints',
        action='store_true',
        help='Clean up old checkpoints (keep best and latest)'
    )
    
    checkpoint_group.add_argument(
        '--merge-checkpoints',
        nargs='+',
        metavar='CHECKPOINT',
        help='Merge multiple checkpoints (experimental)'
    )
    
    # === EXPERIMENT MANAGEMENT ===
    experiment_group = parser.add_argument_group('Experiment Management')
    experiment_group.add_argument(
        '--list-experiments',
        action='store_true',
        help='List all experiments'
    )
    
    experiment_group.add_argument(
        '--archive-experiment',
        type=str,
        metavar='NAME',
        help='Archive an experiment'
    )
    
    experiment_group.add_argument(
        '--compare-experiments',
        nargs='+',
        metavar='NAME',
        help='Compare multiple experiments'
    )
    
    # === MONITORING AND LOGGING ===
    monitoring_group = parser.add_argument_group('Monitoring')
    monitoring_group.add_argument(
        '--tensorboard',
        action='store_true',
        help='Enable TensorBoard logging'
    )
    
    monitoring_group.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    
    monitoring_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    monitoring_group.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (can be used multiple times)'
    )
    
    # === ADVANCED FEATURES ===
    advanced_group = parser.add_argument_group('Advanced Features')
    advanced_group.add_argument(
        '--auto-tune',
        action='store_true',
        help='Enable automatic hyperparameter tuning'
    )
    
    advanced_group.add_argument(
        '--distributed-backend',
        choices=['nccl', 'gloo', 'mpi'],
        help='Distributed training backend'
    )
    
    advanced_group.add_argument(
        '--mixed-precision',
        choices=['fp16', 'bf16', 'fp32'],
        help='Mixed precision training mode'
    )
    
    advanced_group.add_argument(
        '--compile-model',
        action='store_true',
        help='Use torch.compile for model optimization'
    )
    
    return parser


def find_checkpoint_path(checkpoint_spec: str, experiment_name: str = None) -> Optional[Path]:
    """Find checkpoint path based on specification."""
    if checkpoint_spec in ['latest', 'best']:
        # Look in current experiment directory first
        if experiment_name:
            exp_dir = Path(f"experiments/{experiment_name}")
            if checkpoint_spec == 'latest':
                latest_file = exp_dir / "latest_checkpoint.pt"
                if latest_file.exists():
                    return latest_file
            elif checkpoint_spec == 'best':
                best_file = exp_dir / "best_checkpoint.pt"
                if best_file.exists():
                    return best_file
        
        # Look in checkpoints directory
        checkpoints_dir = Path("checkpoints")
        if checkpoint_spec == 'latest':
            # Find most recent checkpoint
            if checkpoints_dir.exists():
                checkpoints = list(checkpoints_dir.glob("checkpoint_*.pt"))
                if checkpoints:
                    return max(checkpoints, key=lambda p: p.stat().st_mtime)
        elif checkpoint_spec == 'best':
            best_file = checkpoints_dir / "best_checkpoint.pt"
            if best_file.exists():
                return best_file
    else:
        # Specific path
        checkpoint_path = Path(checkpoint_spec)
        if checkpoint_path.exists():
            return checkpoint_path
        
        # Try relative to checkpoints directory
        checkpoint_path = Path("checkpoints") / checkpoint_spec
        if checkpoint_path.exists():
            return checkpoint_path
    
    return None


def list_available_checkpoints():
    """List all available checkpoints."""
    print("\n" + "="*60)
    print("AVAILABLE CHECKPOINTS")
    print("="*60)
    
    checkpoints_dir = Path("checkpoints")
    experiments_dir = Path("experiments")
    
    # System-wide checkpoints
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*.pt"))
        if checkpoints:
            print("\nSystem Checkpoints:")
            for ckpt in sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True):
                size_mb = ckpt.stat().st_size / (1024*1024)
                mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
                print(f"  {ckpt.name:<40} {size_mb:8.1f}MB  {mtime.strftime('%Y-%m-%d %H:%M')}")
    
    # Experiment checkpoints
    if experiments_dir.exists():
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                checkpoints = list(exp_dir.glob("*.pt"))
                if checkpoints:
                    print(f"\nExperiment '{exp_dir.name}':")
                    for ckpt in sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True):
                        size_mb = ckpt.stat().st_size / (1024*1024)
                        mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
                        print(f"  {ckpt.name:<40} {size_mb:8.1f}MB  {mtime.strftime('%Y-%m-%d %H:%M')}")


def analyze_model_checkpoint(checkpoint_path: Path):
    """Analyze a model checkpoint."""
    print(f"\n" + "="*60)
    print(f"ANALYZING CHECKPOINT: {checkpoint_path}")
    print("="*60)
    
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Basic info
        print(f"\nCheckpoint Information:")
        print(f"  File size: {checkpoint_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"  Created: {datetime.fromtimestamp(checkpoint_path.stat().st_mtime)}")
        
        if 'save_time' in checkpoint_data:
            print(f"  Saved: {checkpoint_data['save_time']}")
        
        # Training progress
        if 'current_epoch' in checkpoint_data:
            print(f"  Epoch: {checkpoint_data['current_epoch']}")
        if 'global_step' in checkpoint_data:
            print(f"  Step: {checkpoint_data['global_step']}")
        
        # Model architecture
        if 'model_config' in checkpoint_data:
            config = checkpoint_data['model_config']
            print(f"\nModel Architecture:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        
        # Model parameters
        if 'model_state_dict' in checkpoint_data:
            state_dict = checkpoint_data['model_state_dict']
            total_params = sum(p.numel() for p in state_dict.values())
            trainable_params = sum(p.numel() for p in state_dict.values() if (p.requires_grad if hasattr(p, 'requires_grad') else True))
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
            # Memory estimation
            param_memory = sum(p.numel() * p.element_size() for p in state_dict.values() if hasattr(p, 'element_size')) / (1024*1024)
            print(f"  Parameter memory: {param_memory:.1f} MB")
        
        # Training metrics
        if 'metrics' in checkpoint_data:
            metrics = checkpoint_data['metrics']
            print(f"\nTraining Metrics:")
            for key, value in metrics.items():
                if isinstance(value, list) and value:
                    if key.endswith('_losses'):
                        print(f"  {key}: {value[-1]:.6f} (latest)")
                    else:
                        print(f"  {key}: {value[-1]} (latest)")
                else:
                    print(f"  {key}: {value}")
        
        # Configuration
        if 'config' in checkpoint_data:
            config = checkpoint_data['config']
            print(f"\nTraining Configuration:")
            important_keys = ['learning_rate', 'batch_size', 'gradient_accumulation_steps', 'use_moe', 'num_experts']
            for key in important_keys:
                if key in config:
                    print(f"  {key}: {config[key]}")
    
    except Exception as e:
        print(f"Error analyzing checkpoint: {e}")


def create_data_shards(input_file: Path, num_shards: int):
    """Split data into shards for distributed training."""
    print(f"\nCreating {num_shards} data shards from {input_file}")
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return
    
    output_dir = input_file.parent / "shards"
    output_dir.mkdir(exist_ok=True)
    
    # Count total lines first
    total_lines = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    lines_per_shard = max(1, total_lines // num_shards)
    
    print(f"Total lines: {total_lines:,}")
    print(f"Lines per shard: {lines_per_shard:,}")
    
    shard_files = []
    current_shard = 0
    current_lines = 0
    current_file = None
    
    with open(input_file, 'r', encoding='utf-8') as input_f:
        for line in input_f:
            if current_file is None or current_lines >= lines_per_shard:
                if current_file is not None:
                    current_file.close()
                
                shard_path = output_dir / f"{input_file.stem}_shard_{current_shard:04d}.jsonl"
                current_file = open(shard_path, 'w', encoding='utf-8')
                shard_files.append(shard_path)
                current_shard += 1
                current_lines = 0
            
            current_file.write(line)
            current_lines += 1
    
    if current_file is not None:
        current_file.close()
    
    print(f"Created {len(shard_files)} shards in {output_dir}")
    for shard_file in shard_files:
        size_mb = shard_file.stat().st_size / (1024*1024)
        print(f"  {shard_file.name}: {size_mb:.1f} MB")


def benchmark_training_performance(config):
    """Benchmark training performance."""
    print("\n" + "="*60)
    print("BENCHMARKING TRAINING PERFORMANCE")
    print("="*60)
    
    try:
        # Initialize components
        tokenizer = ConversationTokenizer()
        model_config = config_to_deepseek_config(config)
        model = DeepSeekTransformer(model_config)
        
        # Create dummy data
        dummy_data = []
        for i in range(10):
            conversation = {
                "messages": [
                    {"role": "user", "content": f"Test question {i} " * 50},
                    {"role": "assistant", "content": f"Test response {i} " * 50}
                ]
            }
            dummy_data.append(conversation)
        
        # Benchmark tokenization
        start_time = time.time()
        tokenized_data = []
        for conv in dummy_data:
            tokens = tokenizer.encode_conversation(conv)
            tokenized_data.append(tokens)
        tokenization_time = time.time() - start_time
        
        print(f"Tokenization Performance:")
        print(f"  Time: {tokenization_time:.3f}s for {len(dummy_data)} conversations")
        print(f"  Speed: {len(dummy_data) / tokenization_time:.1f} conversations/sec")
        
        # Benchmark model forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create batch
        batch_size = config.batch_size
        seq_len = config.seq_length
        dummy_input = torch.randint(0, min(tokenizer.get_vocab_size(), 1000), (batch_size, seq_len), device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        num_runs = 10
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        forward_time = time.time() - start_time
        
        print(f"\nModel Forward Pass Performance:")
        print(f"  Time: {forward_time:.3f}s for {num_runs} runs")
        print(f"  Average: {forward_time / num_runs * 1000:.1f}ms per forward pass")
        print(f"  Throughput: {batch_size * seq_len * num_runs / forward_time:.0f} tokens/sec")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024*1024*1024)
            print(f"  GPU Memory: {memory_used:.1f} GB")
        
    except Exception as e:
        print(f"Benchmark error: {e}")
        traceback.print_exc()


def export_model_for_inference(checkpoint_path: Path, output_dir: Path, format_type: str):
    """Export model for inference."""
    print(f"\n" + "="*60)
    print(f"EXPORTING MODEL: {format_type.upper()}")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Recreate model
        if 'model_config' in checkpoint_data:
            model_config = checkpoint_data['model_config']
            # Convert to DeepSeekConfig if needed
            from Main import config_to_deepseek_config
            
            # Create a temporary config object
            class TempConfig:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            
            temp_config = TempConfig(**model_config)
            deepseek_config = config_to_deepseek_config(temp_config)
            model = DeepSeekTransformer(deepseek_config)
            
            # Load weights
            model.load_state_dict(checkpoint_data['model_state_dict'])
            model.eval()
            
            print(f"Model loaded successfully")
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            if format_type == 'pytorch':
                # Save PyTorch model
                output_path = output_dir / "model.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': model_config,
                    'tokenizer_config': {},  # Add tokenizer config if available
                }, output_path)
                
                # Save config separately
                config_path = output_dir / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(model_config, f, indent=2)
                
                print(f"PyTorch model saved to: {output_path}")
                print(f"Config saved to: {config_path}")
            
            elif format_type == 'huggingface':
                # Save in HuggingFace format
                try:
                    from transformers import AutoModel, AutoConfig
                    
                    # This would require creating a HuggingFace-compatible model
                    print("HuggingFace export not implemented yet")
                    
                except ImportError:
                    print("HuggingFace transformers not available")
            
            else:
                print(f"Export format '{format_type}' not implemented")
        
        else:
            print("Error: Checkpoint missing model configuration")
    
    except Exception as e:
        print(f"Export error: {e}")
        traceback.print_exc()


def list_experiments():
    """List all experiments with their status."""
    print("\n" + "="*60)
    print("AVAILABLE EXPERIMENTS")
    print("="*60)
    
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        print("No experiments directory found.")
        return
    
    experiments = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            exp_info = {
                'name': exp_dir.name,
                'path': exp_dir,
                'created': datetime.fromtimestamp(exp_dir.stat().st_mtime),
                'checkpoints': len(list(exp_dir.glob("*.pt"))),
                'size_mb': sum(f.stat().st_size for f in exp_dir.rglob("*") if f.is_file()) / (1024*1024)
            }
            
            # Check for summary file
            summary_file = exp_dir / "training_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                        exp_info['status'] = 'completed'
                        exp_info['epochs'] = summary.get('total_epochs', 'unknown')
                        exp_info['final_loss'] = summary.get('final_metrics', {}).get('final_train_loss', 'unknown')
                except:
                    exp_info['status'] = 'unknown'
            else:
                exp_info['status'] = 'in_progress' if exp_info['checkpoints'] > 0 else 'started'
            
            experiments.append(exp_info)
    
    # Sort by creation time
    experiments.sort(key=lambda x: x['created'], reverse=True)
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"{'Name':<30} {'Status':<12} {'Checkpoints':<12} {'Size':<10} {'Created':<16}")
    print("-" * 90)
    
    for exp in experiments:
        print(f"{exp['name']:<30} {exp['status']:<12} {exp['checkpoints']:<12} {exp['size_mb']:8.1f}MB {exp['created'].strftime('%Y-%m-%d %H:%M')}")


def compare_experiments(experiment_names: List[str]):
    """Compare multiple experiments."""
    print("\n" + "="*60)
    print("EXPERIMENT COMPARISON")
    print("="*60)
    
    experiments_data = []
    
    for exp_name in experiment_names:
        exp_dir = Path(f"experiments/{exp_name}")
        if not exp_dir.exists():
            print(f"Experiment '{exp_name}' not found")
            continue
        
        summary_file = exp_dir / "training_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    experiments_data.append({
                        'name': exp_name,
                        'summary': summary
                    })
            except Exception as e:
                print(f"Could not load summary for '{exp_name}': {e}")
        else:
            print(f"No summary found for '{exp_name}'")
    
    if len(experiments_data) < 2:
        print("Need at least 2 experiments to compare")
        return
    
    # Compare key metrics
    print(f"\n{'Metric':<25} " + " ".join(f"{exp['name']:<15}" for exp in experiments_data))
    print("-" * (25 + 16 * len(experiments_data)))
    
    metrics_to_compare = [
        ('Total Training Time (h)', lambda x: x.get('total_training_time_hours', 'N/A')),
        ('Final Train Loss', lambda x: x.get('final_metrics', {}).get('final_train_loss', 'N/A')),
        ('Best Eval Loss', lambda x: x.get('final_metrics', {}).get('best_eval_loss', 'N/A')),
        ('Total Epochs', lambda x: x.get('total_epochs', 'N/A')),
        ('Total Steps', lambda x: x.get('total_steps', 'N/A')),
        ('Avg Throughput', lambda x: x.get('final_metrics', {}).get('avg_throughput', 'N/A')),
    ]
    
    for metric_name, extractor in metrics_to_compare:
        values = []
        for exp in experiments_data:
            value = extractor(exp['summary'])
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        
        print(f"{metric_name:<25} " + " ".join(f"{val:<15}" for val in values))


def clean_old_checkpoints():
    """Clean up old checkpoints keeping only best and latest."""
    print("\n" + "="*60)
    print("CLEANING OLD CHECKPOINTS")
    print("="*60)
    
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        print("No checkpoints directory found.")
        return
    
    # Find all checkpoints
    all_checkpoints = list(checkpoints_dir.glob("checkpoint_*.pt"))
    if not all_checkpoints:
        print("No checkpoints to clean.")
        return
    
    print(f"Found {len(all_checkpoints)} checkpoints")
    
    # Find latest (by modification time)
    latest_checkpoint = max(all_checkpoints, key=lambda p: p.stat().st_mtime)
    
    # Find best checkpoint (if exists)
    best_checkpoint = checkpoints_dir / "best_checkpoint.pt"
    
    # Checkpoints to keep
    keep_checkpoints = {latest_checkpoint}
    if best_checkpoint.exists():
        keep_checkpoints.add(best_checkpoint)
    
    # Also keep emergency and backup checkpoints
    for checkpoint in all_checkpoints:
        if 'emergency' in checkpoint.name or 'backup' in checkpoint.name:
            keep_checkpoints.add(checkpoint)
    
    # Delete the rest
    deleted_count = 0
    total_size_freed = 0
    
    for checkpoint in all_checkpoints:
        if checkpoint not in keep_checkpoints:
            size_mb = checkpoint.stat().st_size / (1024*1024)
            try:
                checkpoint.unlink()
                deleted_count += 1
                total_size_freed += size_mb
                print(f"  Deleted: {checkpoint.name} ({size_mb:.1f}MB)")
            except Exception as e:
                print(f"  Failed to delete {checkpoint.name}: {e}")
    
    print(f"\nCleaned {deleted_count} checkpoints, freed {total_size_freed:.1f}MB")
    print(f"Kept {len(keep_checkpoints)} important checkpoints:")
    for checkpoint in keep_checkpoints:
        if checkpoint.exists():
            size_mb = checkpoint.stat().st_size / (1024*1024)
            print(f"  {checkpoint.name} ({size_mb:.1f}MB)")


def archive_experiment(experiment_name: str):
    """Archive an experiment to compressed backup."""
    print(f"\n" + "="*60)
    print(f"ARCHIVING EXPERIMENT: {experiment_name}")
    print("="*60)
    
    exp_dir = Path(f"experiments/{experiment_name}")
    if not exp_dir.exists():
        print(f"Experiment '{experiment_name}' not found")
        return
    
    archive_dir = Path("archived_experiments")
    archive_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{experiment_name}_{timestamp}"
    
    try:
        import shutil
        
        # Create compressed archive
        archive_path = archive_dir / archive_name
        shutil.make_archive(str(archive_path), 'zip', str(exp_dir))
        
        archive_file = Path(f"{archive_path}.zip")
        size_mb = archive_file.stat().st_size / (1024*1024)
        
        print(f"Experiment archived to: {archive_file}")
        print(f"Archive size: {size_mb:.1f}MB")
        
        # Create archive metadata
        metadata = {
            'original_name': experiment_name,
            'archive_date': datetime.now().isoformat(),
            'original_size_mb': sum(f.stat().st_size for f in exp_dir.rglob("*") if f.is_file()) / (1024*1024),
            'archive_size_mb': size_mb,
            'compression_ratio': f"{(1 - size_mb / max(1, sum(f.stat().st_size for f in exp_dir.rglob('*') if f.is_file()) / (1024*1024))) * 100:.1f}%"
        }
        
        metadata_file = archive_dir / f"{archive_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Archive metadata saved to: {metadata_file}")
        
        # Ask if user wants to delete original
        response = input(f"\nDelete original experiment directory? (y/N): ").strip().lower()
        if response == 'y':
            shutil.rmtree(exp_dir)
            print(f"Original experiment directory deleted")
        
    except Exception as e:
        print(f"Archive error: {e}")
        traceback.print_exc()


def test_deepspeed_setup(config):
    """Test DeepSpeed configuration and setup."""
    print("\n" + "="*60)
    print("TESTING DEEPSPEED SETUP")
    print("="*60)
    
    if not DEEPSPEED_AVAILABLE:
        print("DeepSpeed not available - cannot test setup")
        return
    
    # Check environment variables
    print("Environment Variables:")
    env_vars = ['WORLD_SIZE', 'LOCAL_RANK', 'RANK', 'MASTER_ADDR', 'MASTER_PORT']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Test basic DeepSpeed import and version
    try:
        import deepspeed
        print(f"\nDeepSpeed version: {deepspeed.__version__}")
    except Exception as e:
        print(f"DeepSpeed import error: {e}")
        return
    
    # Test model creation
    try:
        print("\nTesting model creation...")
        tokenizer = ConversationTokenizer()
        model_config = config_to_deepseek_config(config)
        model = DeepSeekTransformer(model_config)
        print(f"Model created successfully: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"Model creation error: {e}")
        return
    
    # Test DeepSpeed config generation
    try:
        print("\nTesting DeepSpeed configuration...")
        from training.trainer import EnhancedConversationTrainer
        
        # Create a temporary trainer to test config generation
        trainer = EnhancedConversationTrainer(model, tokenizer, config, logging.getLogger())
        ds_config = trainer._create_deepspeed_config()
        
        print("DeepSpeed configuration generated successfully")
        print(f"  Train batch size: {ds_config.get('train_batch_size', 'N/A')}")
        print(f"  Micro batch size: {ds_config.get('train_micro_batch_size_per_gpu', 'N/A')}")
        print(f"  ZeRO stage: {ds_config.get('zero_optimization', {}).get('stage', 'N/A')}")
        print(f"  CPU offload: {ds_config.get('zero_optimization', {}).get('offload_optimizer', {}).get('device', 'none') == 'cpu'}")
        
    except Exception as e:
        print(f"DeepSpeed configuration error: {e}")
        traceback.print_exc()
        return
    
    # Test DeepSpeed initialization (dry run)
    try:
        print("\nTesting DeepSpeed initialization (dry run)...")
        
        # This would normally initialize DeepSpeed, but we'll just validate the config
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if world_size > 1:
            print(f"Distributed setup detected: {world_size} processes")
        else:
            print("Single process setup")
        
        print("DeepSpeed setup test completed successfully")
        
    except Exception as e:
        print(f"DeepSpeed initialization test error: {e}")
        traceback.print_exc()


def profile_memory_usage(config):
    """Profile memory usage during model operations."""
    print("\n" + "="*60)
    print("MEMORY PROFILING")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available - limited memory profiling")
    
    def print_memory_stats(stage: str):
        """Print current memory statistics."""
        print(f"\n{stage}:")
        
        # System memory
        memory = psutil.virtual_memory()
        print(f"  System Memory: {memory.used / (1024**3):.1f}GB used / {memory.total / (1024**3):.1f}GB total ({memory.percent:.1f}%)")
        
        # GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"  GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        
        # Python process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024**3)
        print(f"  Process Memory: {process_memory:.1f}GB")
    
    print_memory_stats("Initial Memory State")
    
    # Create tokenizer
    print("\nCreating tokenizer...")
    tokenizer = ConversationTokenizer()
    print_memory_stats("After Tokenizer Creation")
    
    # Create model
    print("\nCreating model...")
    model_config = config_to_deepseek_config(config)
    model = DeepSeekTransformer(model_config)
    print_memory_stats("After Model Creation")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        print("\nMoving model to GPU...")
        model = model.to('cuda')
        print_memory_stats("After Moving to GPU")
    
    # Create optimizer
    print("\nCreating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    print_memory_stats("After Optimizer Creation")
    
    # Test forward pass
    print("\nTesting forward pass...")
    device = next(model.parameters()).device
    batch_size = min(config.batch_size, 2)  # Reduce batch size for profiling
    seq_len = min(config.seq_length, 1024)  # Reduce sequence length
    
    dummy_input = torch.randint(0, min(tokenizer.get_vocab_size(), 1000), (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        output = model(dummy_input)
    print_memory_stats("After Forward Pass")
    
    # Test backward pass
    print("\nTesting backward pass...")
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    
    # Simple loss for backward pass
    labels = dummy_input
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    loss.backward()
    print_memory_stats("After Backward Pass")
    
    # Optimizer step
    print("\nTesting optimizer step...")
    optimizer.step()
    optimizer.zero_grad()
    print_memory_stats("After Optimizer Step")
    
    # Cleanup
    print("\nCleaning up...")
    del model, optimizer, dummy_input, output, logits, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_memory_stats("After Cleanup")


def safe_model_memory_footprint(model):
    """Get model memory footprint with comprehensive error handling."""
    try:
        # Try the model's built-in method first
        if hasattr(model, 'get_memory_footprint'):
            return model.get_memory_footprint()
    except Exception as e:
        print(f"Built-in memory footprint calculation failed: {e}")
    
    # Fallback calculation
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'total_size_mb': total_size / (1024 * 1024),
            'method': 'fallback_calculation'
        }
    except Exception as e:
        print(f"Fallback memory calculation also failed: {e}")
        return {
            'total_parameters': 'unknown',
            'total_size_mb': 'unknown',
            'error': str(e)
        }


def benchmark_actual_performance(model, tokenizer, config, dist_manager, samples=20):
    """Measure real training performance for accurate time estimation."""
    print("MEASURING ACTUAL PERFORMANCE...")
    
    # Create realistic test data
    test_inputs = []
    for i in range(samples):
        # Create varied length test sequences
        base_length = config.seq_length // 4 + (i * 10)
        test_text = f"Training benchmark sample {i}. This is a realistic test sequence. " * (base_length // 50)
        
        try:
            tokens = tokenizer.encode_conversation({"messages": [
                {"role": "user", "content": test_text}
            ]})
            if len(tokens) > config.seq_length:
                tokens = tokens[:config.seq_length]
            else:
                tokens.extend([tokenizer.pad_token_id] * (config.seq_length - len(tokens)))
            
            test_inputs.append(torch.tensor(tokens, dtype=torch.long))
        except Exception as e:
            print(f"Error creating test input {i}: {e}")
            tokens = torch.randint(0, min(tokenizer.get_vocab_size(), 1000), (config.seq_length,), dtype=torch.long)
            test_inputs.append(tokens)
    
    # Create test dataloader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_inputs, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Setup for measurement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = model.to(device)
        model.train()
    except Exception as e:
        print(f"Could not move model to device: {e}")
        device = torch.device('cpu')
        model = model.to(device)
    
    # Create optimizer matching training config
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=getattr(config, 'weight_decay', 0.01)
        )
    except Exception as e:
        print(f"Could not create optimizer: {e}")
        return {
            'single_step_time': 1.0,
            'tokens_per_step': 100,
            'error': str(e)
        }
    
    # Warmup runs (don't count these)
    print("Warming up...")
    warmup_batches = min(3, len(test_loader))
    for i, batch in enumerate(test_loader):
        if i >= warmup_batches:
            break
            
        try:
            batch = batch.to(device, non_blocking=True)
            attention_mask = (batch != tokenizer.pad_token_id).long() if hasattr(tokenizer, 'pad_token_id') else torch.ones_like(batch)
            
            # Simple forward pass
            outputs = model(batch, attention_mask)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Calculate loss manually
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        except Exception as e:
            print(f"Warmup error: {e}")
            continue
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Clear memory stats and synchronize
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # ACTUAL MEASUREMENT
    print("Measuring training performance...")
    measurement_start = time.time()
    
    total_tokens_processed = 0
    total_steps = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 5:  # Limit measurement batches
            break
            
        step_start = time.time()
        
        try:
            batch = batch.to(device, non_blocking=True)
            batch_size, seq_len = batch.shape
            
            attention_mask = (batch != tokenizer.pad_token_id).long() if hasattr(tokenizer, 'pad_token_id') else torch.ones_like(batch)
            
            # Forward pass
            outputs = model(batch, attention_mask)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Calculate loss manually
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation simulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Track metrics
            total_tokens_processed += batch_size * seq_len
            total_samples += batch_size
            total_steps += 1
            
        except Exception as e:
            print(f"Measurement error on batch {batch_idx}: {e}")
            continue
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    measurement_end = time.time()
    total_measurement_time = measurement_end - measurement_start
    
    # Calculate per-step performance
    single_step_time = total_measurement_time / max(total_steps, 1)
    tokens_per_step = total_tokens_processed / max(total_steps, 1)
    samples_per_step = total_samples / max(total_steps, 1)
    
    # Memory statistics
    peak_memory_gb = 0
    current_memory_gb = 0
    if torch.cuda.is_available():
        try:
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            current_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        except:
            pass
    
    performance_stats = {
        'single_step_time': single_step_time,
        'tokens_per_step': tokens_per_step,
        'samples_per_step': samples_per_step,
        'total_time': total_measurement_time,
        'tokens_processed': total_tokens_processed,
        'samples_processed': total_samples,
        'steps_completed': total_steps,
        'peak_memory_gb': peak_memory_gb,
        'current_memory_gb': current_memory_gb,
        'world_size': dist_manager.world_size
    }
    
    print(f"Performance measurement complete!")
    print(f"   Measured {total_steps} steps in {total_measurement_time:.1f}s")
    print(f"   Single step time: {single_step_time:.3f}s")
    print(f"   Tokens per step: {tokens_per_step:,.0f}")
    print(f"   Memory usage: {current_memory_gb:.1f}GB (peak: {peak_memory_gb:.1f}GB)")
    
    return performance_stats


class DeepSpeedIntegrationManager:
    """Manages DeepSpeed integration and distributed training setup."""
    
    def __init__(self):
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.rank = int(os.environ.get('RANK', 0))
        self.is_distributed = self.world_size > 1
        
    def setup_distributed_environment(self):
        """Setup distributed training environment."""
        if not self.is_distributed:
            logging.info("Single GPU/CPU training mode")
            return
        
        logging.info(f"Distributed training setup:")
        logging.info(f"  World size: {self.world_size}")
        logging.info(f"  Global rank: {self.rank}")
        logging.info(f"  Local rank: {self.local_rank}")
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            
        # Initialize process group for non-DeepSpeed communication if needed
        if not DEEPSPEED_AVAILABLE:
            try:
                import torch.distributed as dist
                dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
                logging.info("PyTorch distributed initialized")
            except Exception as e:
                logging.error(f"Failed to initialize PyTorch distributed: {e}")
    
    def should_log(self) -> bool:
        """Check if this process should log (only rank 0)."""
        return self.rank == 0
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information for this process."""
        info = {
            'rank': self.rank,
            'local_rank': self.local_rank,
            'world_size': self.world_size,
            'is_distributed': self.is_distributed
        }
        
        if torch.cuda.is_available():
            try:
                info.update({
                    'device': f'cuda:{self.local_rank}',
                    'gpu_name': torch.cuda.get_device_name(self.local_rank),
                    'gpu_memory_gb': torch.cuda.get_device_properties(self.local_rank).total_memory / 1e9
                })
            except Exception as e:
                info['device'] = 'cuda'
                info['gpu_error'] = str(e)
        else:
            info['device'] = 'cpu'
        
        return info


class EnhancedResourceManager:
    """Enhanced resource management with DeepSpeed optimizations."""
    
    def __init__(self, distributed_manager: DeepSpeedIntegrationManager):
        self.distributed_manager = distributed_manager
        self.system_info = self._gather_comprehensive_system_info()
        
    def _gather_comprehensive_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information across all nodes."""
        info = {
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'distributed_info': self.distributed_manager.get_device_info()
        }
        
        # GPU information per device
        if torch.cuda.is_available():
            info['gpu_info'] = []
            try:
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    info['gpu_info'].append({
                        'device_id': i,
                        'name': gpu_props.name,
                        'memory_gb': gpu_props.total_memory / 1e9,
                        'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                        'multiprocessor_count': gpu_props.multi_processor_count
                    })
            except Exception as e:
                info['gpu_info'] = [{'error': str(e)}]
        
        # Storage information
        try:
            disk_usage = psutil.disk_usage('.')
            info['disk_free_gb'] = disk_usage.free / (1024**3)
            info['disk_total_gb'] = disk_usage.total / (1024**3)
        except Exception:
            info.update({'disk_free_gb': 100, 'disk_total_gb': 500})
        
        return info
    
    def create_deepspeed_optimization_strategy(self, config: Config, model_size_gb: float, 
                                             sequence_length: int) -> Dict[str, Any]:
        """Create comprehensive DeepSpeed optimization strategy."""
        strategy = {
            'use_deepspeed': DEEPSPEED_AVAILABLE and getattr(config, 'use_deepspeed', False),
            'zero_stage': 3,  # Default to ZeRO-3
            'cpu_offload': False,
            'nvme_offload': False,
            'moe_optimizations': {},
            'precision_strategy': 'bf16',
            'gradient_accumulation_strategy': {},
            'communication_optimizations': {}
        }
        
        if not strategy['use_deepspeed']:
            return strategy
        
        # Analyze memory requirements
        total_gpu_memory = sum(gpu.get('memory_gb', 0) for gpu in self.system_info.get('gpu_info', []))
        available_cpu_memory = self.system_info['available_memory_gb']
        
        # Memory-based decisions
        memory_pressure_ratio = model_size_gb / max(total_gpu_memory, 1.0)
        
        if memory_pressure_ratio > 0.8:  # High memory pressure
            strategy.update({
                'zero_stage': 3,
                'cpu_offload': True,
                'aggressive_cpu_offload': True,
                'precision_strategy': 'fp16',  # More memory efficient
                'activation_checkpointing': True
            })
            logging.info("High memory pressure detected - enabling aggressive optimizations")
            
        elif memory_pressure_ratio > 0.5:  # Moderate memory pressure
            strategy.update({
                'zero_stage': 3,
                'cpu_offload': True,
                'precision_strategy': 'bf16'
            })
            logging.info("Moderate memory pressure - enabling CPU offload")
            
        else:  # Low memory pressure
            strategy.update({
                'zero_stage': 2,  # Less aggressive
                'precision_strategy': 'bf16'
            })
            logging.info("Low memory pressure - using standard ZeRO-2")
        
        # NVMe offloading if path is provided and memory is very constrained
        nvme_path = getattr(config, 'nvme_path', None)
        if nvme_path and memory_pressure_ratio > 1.0:
            strategy['nvme_offload'] = True
            strategy['nvme_path'] = nvme_path
            logging.info(f"Enabling NVMe offload to {nvme_path}")
        
        # MoE-specific optimizations
        if hasattr(config, 'use_moe') and config.use_moe and DeepSpeedConfigGenerator:
            num_experts = getattr(config, 'num_experts', 8)
            world_size = self.distributed_manager.world_size
            
            try:
                moe_config = DeepSpeedConfigGenerator.create_moe_config_with_expert_parallelism(
                    world_size, num_experts, model_size_gb, sequence_length
                )
                strategy['moe_optimizations'] = moe_config['moe']
            except Exception as e:
                logging.warning(f"MoE configuration failed: {e}")
            
            # Adjust other settings for MoE
            if sequence_length > 100000:  # Very long sequences with MoE
                strategy['gradient_accumulation_strategy'] = {
                    'recommended_micro_batch_size': 1,
                    'recommended_gradient_accumulation': max(config.batch_size * 8, 64),
                    'reason': 'Long sequences with MoE require small micro batches'
                }
        
        # Communication optimizations for multi-node
        if self.distributed_manager.world_size > 8:  # Multi-node likely
            strategy['communication_optimizations'] = {
                'overlap_comm': True,
                'allgather_bucket_size': 5e8,
                'reduce_bucket_size': 5e8,
                'communication_data_type': strategy['precision_strategy'],
                'contiguous_gradients': True
            }
        
        return strategy
    
    def _estimate_model_memory_usage(self, config: Config) -> float:
        """Estimate model memory usage in GB."""
        try:
            # Basic parameter count
            hidden_size = getattr(config, 'hidden_size', 768)
            num_layers = getattr(config, 'num_layers', 12)
            vocab_size = getattr(config, 'vocab_size', 50257)
            
            # Rough parameter estimation
            params = (
                vocab_size * hidden_size +  # Embeddings
                num_layers * (
                    hidden_size * hidden_size * 4 +  # Attention
                    hidden_size * getattr(config, 'intermediate_size', hidden_size * 4) * 2  # FFN
                )
            )
            
            # Account for MoE if enabled
            if hasattr(config, 'use_moe') and config.use_moe:
                num_experts = getattr(config, 'num_experts', 8)
                params += num_layers * num_experts * hidden_size * getattr(config, 'intermediate_size', hidden_size * 4)
            
            # Memory usage: parameters + gradients + optimizer states + activations
            # Parameters: 4 bytes per param (fp32) or 2 bytes (fp16/bf16)
            param_memory = params * 2 / 1e9  # Assume mixed precision
            
            # Gradients: same size as parameters
            grad_memory = param_memory
            
            # Optimizer states (Adam): 8 bytes per parameter
            optimizer_memory = params * 8 / 1e9
            
            # Activations (rough estimate based on sequence length and batch size)
            batch_size = getattr(config, 'batch_size', 8)
            seq_length = getattr(config, 'seq_length', 2048)
            activation_memory = batch_size * seq_length * hidden_size * num_layers * 4 / 1e9
            
            total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
            
            return total_memory
            
        except Exception as e:
            logging.debug(f"Memory estimation failed: {e}")
            return 10.0  # Conservative default


class AdaptiveDeepSpeedWizard:
    """Enhanced configuration wizard with DeepSpeed integration."""
    
    def __init__(self, resource_manager: EnhancedResourceManager):
        self.resource_manager = resource_manager
    
    def auto_configure_deepspeed(self, config: Config, 
                                natural_description: Optional[str] = None,
                                manual_overrides: Dict[str, Any] = None) -> Config:
        """Automatically configure DeepSpeed based on system analysis, with proper override handling."""
        
        # Apply manual overrides BEFORE auto-configuration
        if manual_overrides:
            print("="*60)
            print("APPLYING MANUAL CONFIGURATION OVERRIDES")
            print("="*60)
            
            for param, value in manual_overrides.items():
                if hasattr(config, param):
                    old_value = getattr(config, param)
                    setattr(config, param, value)
                    print(f"OVERRIDE: {param}: {old_value} -> {value}")
                else:
                    # Create new attribute for DeepSpeed-specific settings
                    setattr(config, param, value)
                    print(f"NEW ATTRIBUTE: {param} = {value}")
            
            print("="*60)
        
        # Estimate model size from config
        try:
            model_size_gb = self.resource_manager._estimate_model_memory_usage(config)
            sequence_length = getattr(config, 'seq_length', 2048)
        except Exception as e:
            logging.warning(f"Could not estimate model size: {e}")
            model_size_gb = 10.0  # Default estimate
            sequence_length = 2048
        
        # Analyze system and create strategy (ONLY if no manual overrides for these specific params)
        strategy = self.resource_manager.create_deepspeed_optimization_strategy(
            config, model_size_gb, sequence_length
        )
        
        # Apply optimizations to config, but RESPECT manual overrides
        apply_deepspeed_optimizations(config, strategy)
        
        # RESTORE manual overrides if they were set
        if manual_overrides:
            for param, value in manual_overrides.items():
                if hasattr(config, param):
                    setattr(config, param, value)
                    print(f"PRESERVED MANUAL OVERRIDE: {param} = {value}")
        
        # Log final strategy
        print("="*60)
        print("FINAL DEEPSPEED CONFIGURATION")
        print("="*60)
        print(f"Model size estimate: {model_size_gb:.1f}GB")
        print(f"ZeRO stage: {getattr(config, 'zero_stage', 'N/A')}")
        print(f"CPU offload: {getattr(config, 'cpu_offload', False)}")
        print(f"CPU offload optimizer: {getattr(config, 'cpu_offload_optimizer', False)}")
        print(f"Precision: {getattr(config, 'precision', 'auto')}")
        print(f"Use DeepSpeed: {getattr(config, 'use_deepspeed', False)}")
        print("="*60)
        
        return config


def apply_deepspeed_optimizations(config: Config, strategy: Dict[str, Any]):
    """Apply DeepSpeed optimization strategy to config."""
    
    # Enable DeepSpeed
    config.use_deepspeed = strategy.get('use_deepspeed', True)
    
    # ZeRO configuration
    if not hasattr(config, 'zero_stage') or config.zero_stage == 0:
        config.zero_stage = strategy.get('zero_stage', 3)
    
    # CPU offloading
    if not hasattr(config, 'cpu_offload'):
        config.cpu_offload = strategy.get('cpu_offload', False)
        config.cpu_offload_optimizer = strategy.get('cpu_offload', False)
        config.aggressive_cpu_offload = strategy.get('aggressive_cpu_offload', False)
    
    # NVMe offloading
    if strategy.get('nvme_offload', False):
        config.nvme_path = strategy.get('nvme_path')
    
    # Precision strategy
    if strategy.get('precision_strategy') and not hasattr(config, 'precision'):
        config.precision = strategy['precision_strategy']
    
    # Gradient accumulation adjustments
    grad_strategy = strategy.get('gradient_accumulation_strategy', {})
    if grad_strategy:
        if 'recommended_micro_batch_size' in grad_strategy:
            config.batch_size = grad_strategy['recommended_micro_batch_size']
        if 'recommended_gradient_accumulation' in grad_strategy:
            config.gradient_accumulation_steps = grad_strategy['recommended_gradient_accumulation']
        logging.info(f"Adjusted batch size to {config.batch_size}, grad accumulation to {config.gradient_accumulation_steps}")
        logging.info(f"Reason: {grad_strategy.get('reason', 'DeepSpeed optimization')}")
    
    # MoE optimizations
    moe_opts = strategy.get('moe_optimizations', {})
    if moe_opts:
        # Apply MoE-specific settings
        for key, value in moe_opts.items():
            if hasattr(config, key):
                setattr(config, key, value)
        logging.info("Applied MoE optimizations to configuration")


def create_enhanced_directory_structure():
    """Create comprehensive directory structure including DeepSpeed."""
    directories = [
        'data', 'data/shards', 'data/processed', 'data/cache',
        'checkpoints', 'checkpoints/best', 'checkpoints/emergency', 'checkpoints/deepspeed',
        'experiments', 'experiments/archive',
        'logs', 'logs/adaptive', 'logs/performance', 'logs/deepspeed',
        'backups', 'backups/configs', 'backups/models',
        'reports', 'reports/adaptive', 'reports/performance', 'reports/moe',
        'monitoring', 'monitoring/metrics', 'monitoring/visualizations', 'monitoring/routing'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)


def setup_deepspeed_logging():
    """Setup enhanced logging for DeepSpeed training."""
    log_dir = Path('logs/deepspeed')
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create separate log files for different components
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.FileHandler(log_dir / f'moe_routing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=handlers
    )
    
    # Set DeepSpeed logging level
    if DEEPSPEED_AVAILABLE:
        deepspeed_logger = logging.getLogger('deepspeed')
        deepspeed_logger.setLevel(logging.INFO)


def config_to_deepseek_config(config: Config):
    """Convert training Config to DeepSeekConfig with MoE support."""
    return DeepSeekConfig(
        vocab_size=getattr(config, 'vocab_size', 50257),
        hidden_size=getattr(config, 'hidden_size', 768),
        num_layers=getattr(config, 'num_layers', 12),
        num_heads=getattr(config, 'num_heads', 12),
        num_kv_heads=getattr(config, 'num_kv_heads', None),
        intermediate_size=getattr(config, 'intermediate_size', None),
        seq_length=getattr(config, 'seq_length', 2048),
        dropout=getattr(config, 'dropout', 0.0),
        rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
        rope_theta=getattr(config, 'rope_theta', 10000.0),
        init_std=getattr(config, 'init_std', 0.02),
        use_stable_embedding=getattr(config, 'use_stable_embedding', True),
        tie_word_embeddings=getattr(config, 'tie_word_embeddings', True),
        gradient_checkpointing=getattr(config, 'gradient_checkpointing', False),
        
        # MoE configuration
        use_moe=getattr(config, 'use_moe', False),
        num_experts=getattr(config, 'num_experts', 8),
        moe_top_k=getattr(config, 'moe_top_k', 2),
        capacity_factor=getattr(config, 'capacity_factor', 1.25),
        load_balancing_weight=getattr(config, 'load_balancing_weight', 0.01),
    )


def create_dummy_training_data(file_path: Path, num_samples: int = 50):
    """Create dummy training data for testing."""
    print(f"Creating dummy training data: {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    dummy_conversations = []
    for i in range(num_samples):
        conversation = {
            "messages": [
                {"role": "user", "content": f"This is test question number {i+1}. Can you help me with a simple task?"},
                {"role": "assistant", "content": f"Of course! I'd be happy to help you with test response {i+1}. This is a sample response for training purposes."}
            ]
        }
        dummy_conversations.append(conversation)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for conv in dummy_conversations:
            f.write(json.dumps(conv) + '\n')
    
    print(f"Created {num_samples} dummy conversations in {file_path}")


def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(f"CUDA cleanup error: {e}")


def create_fallback_trainer(model, tokenizer, config):
    """Create a basic fallback trainer if enhanced trainer fails."""
    
    class BasicTrainer:
        def __init__(self, model, tokenizer, config):
            self.model = model
            self.tokenizer = tokenizer
            self.config = config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Try to move model to device
            try:
                self.model = self.model.to(self.device)
            except Exception as e:
                print(f"Could not move model to {self.device}: {e}")
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
            
            # Basic optimizer
            try:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), 
                    lr=config.learning_rate,
                    weight_decay=getattr(config, 'weight_decay', 0.01)
                )
            except Exception as e:
                print(f"Could not create optimizer: {e}")
                self.optimizer = None
        
        def train(self, train_dataset, eval_dataset=None):
            print("="*60)
            print("RUNNING WITH BASIC FALLBACK TRAINER")
            print("="*60)
            print("This is a minimal trainer for testing purposes only")
            print("For full training, fix the enhanced trainer import")
            
            if not self.optimizer:
                print("No optimizer available, cannot train")
                return
            
            try:
                # Create basic dataloader
                from torch.utils.data import DataLoader
                train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
                
                self.model.train()
                
                for epoch in range(min(2, config.num_epochs)):  # Limit epochs
                    print(f"Basic training epoch {epoch + 1}")
                    
                    for batch_idx, batch in enumerate(train_loader):
                        if batch_idx >= 5:  # Limit batches
                            break
                        
                        try:
                            # Move to device
                            if isinstance(batch, dict):
                                batch = {k: v.to(self.device) for k, v in batch.items()}
                                input_ids = batch.get('input_ids')
                                attention_mask = batch.get('attention_mask')
                            else:
                                input_ids = batch.to(self.device)
                                attention_mask = None
                            
                            if input_ids is None:
                                continue
                            
                            # Forward pass
                            outputs = self.model(input_ids, attention_mask)
                            
                            if isinstance(outputs, tuple):
                                logits = outputs[0]
                            else:
                                logits = outputs
                            
                            # Simple loss calculation
                            labels = input_ids
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            
                            loss_fct = torch.nn.CrossEntropyLoss()
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            
                            # Backward pass
                            loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            
                            if batch_idx % 2 == 0:
                                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
                                
                        except Exception as e:
                            print(f"Training step error: {e}")
                            continue
                
                print("Basic training completed")
                
            except Exception as e:
                print(f"Basic training failed: {e}")
                traceback.print_exc()
    
    return BasicTrainer(model, tokenizer, config)


def validate_deepspeed_config(config):
    """Validate DeepSpeed configuration for common issues."""
    issues = []
    
    if getattr(config, 'use_deepspeed', False) and not DEEPSPEED_AVAILABLE:
        issues.append("DeepSpeed requested but not available")
    
    if getattr(config, 'cpu_offload', False) and not getattr(config, 'use_deepspeed', False):
        issues.append("CPU offloading requires DeepSpeed")
    
    if getattr(config, 'zero_stage', 0) > 0 and not getattr(config, 'use_deepspeed', False):
        issues.append("ZeRO stages require DeepSpeed")
    
    # Check batch size configuration
    batch_size = getattr(config, 'batch_size', 1)
    grad_accum = getattr(config, 'gradient_accumulation_steps', 1)
    if batch_size * grad_accum > 64:
        issues.append(f"Very large effective batch size: {batch_size * grad_accum}")
    
    # Check memory requirements
    seq_length = getattr(config, 'seq_length', 2048)
    if seq_length > 8192 and not getattr(config, 'gradient_checkpointing', False):
        issues.append("Long sequences without gradient checkpointing may cause OOM")
    
    return issues


def load_checkpoint_for_resume(checkpoint_path: Path, model, optimizer=None, scheduler=None):
    """Load checkpoint for resuming training."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        if 'model_state_dict' in checkpoint_data:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")  # Show first 5
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint_data:
            try:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                print("Optimizer state loaded")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint_data and checkpoint_data['scheduler_state_dict']:
            try:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                print("Scheduler state loaded")
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")
        
        # Return training state
        current_epoch = checkpoint_data.get('current_epoch', 0)
        global_step = checkpoint_data.get('global_step', 0)
        
        print(f"Resuming from epoch {current_epoch}, step {global_step}")
        
        return {
            'current_epoch': current_epoch,
            'global_step': global_step,
            'metrics': checkpoint_data.get('metrics', {}),
            'config': checkpoint_data.get('config', {})
        }
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        traceback.print_exc()
        return None


def debug_training_setup(config):
    """Debug training setup and configuration."""
    print("\n" + "="*60)
    print("DEBUG TRAINING SETUP")
    print("="*60)
    
    print("Configuration:")
    config_dict = {}
    for attr in dir(config):
        if not attr.startswith('_') and not callable(getattr(config, attr)):
            config_dict[attr] = getattr(config, attr)
    
    for key, value in sorted(config_dict.items()):
        print(f"  {key}: {value}")
    
    print("\nEnvironment:")
    important_env_vars = ['CUDA_VISIBLE_DEVICES', 'WORLD_SIZE', 'LOCAL_RANK', 'RANK', 'MASTER_ADDR', 'MASTER_PORT']
    for var in important_env_vars:
        print(f"  {var}: {os.environ.get(var, 'Not set')}")
    
    print("\nSystem Info:")
    print(f"  Python version: {sys.version}")
    print(f"  PyTorch version: {torch.__version__}")
    if DEEPSPEED_AVAILABLE:
        print(f"  DeepSpeed version: {deepspeed.__version__}")
    else:
        print("  DeepSpeed: Not available")
    
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    # Test component initialization
    print("\nTesting Component Initialization:")
    
    try:
        print("  Creating tokenizer...")
        tokenizer = ConversationTokenizer()
        print(f"    Success: vocab_size={tokenizer.get_vocab_size()}")
    except Exception as e:
        print(f"    Failed: {e}")
    
    try:
        print("  Creating model config...")
        model_config = config_to_deepseek_config(config)
        print(f"    Success: {model_config.hidden_size}d, {model_config.num_layers} layers")
    except Exception as e:
        print(f"    Failed: {e}")
    
    try:
        print("  Creating model...")
        model = DeepSeekTransformer(model_config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"    Success: {param_count:,} parameters")
    except Exception as e:
        print(f"    Failed: {e}")
    
    # Test data loading
    print("\nTesting Data Loading:")
    train_data_path = Path(config.train_data_path)
    if train_data_path.exists():
        print(f"  Training data found: {train_data_path}")
        file_size = train_data_path.stat().st_size / (1024*1024)
        print(f"    Size: {file_size:.1f}MB")
    else:
        print(f"  Training data missing: {train_data_path}")
        print("    Will create dummy data for testing")


def main():
    """Enhanced main function with comprehensive command-line features."""
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    log_levels = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
    logging.basicConfig(level=log_levels.get(args.log_level, logging.INFO))
    
    # Handle special commands that don't require full setup
    if args.list_checkpoints:
        list_available_checkpoints()
        return
    
    if args.list_experiments:
        list_experiments()
        return
    
    if args.compare_experiments:
        compare_experiments(args.compare_experiments)
        return
    
    if args.clean_checkpoints:
        clean_old_checkpoints()
        return
    
    if args.archive_experiment:
        archive_experiment(args.archive_experiment)
        return
    
    if args.create_backup:
        print("Creating backup...")
        backup_dir = Path("backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        
        # Backup important directories
        for source_dir in ['checkpoints', 'experiments', 'logs']:
            if Path(source_dir).exists():
                shutil.copytree(source_dir, backup_dir / source_dir, ignore_errors=True)
        
        print(f"Backup created: {backup_dir}")
        return
    
    # HARDCODED CONFIGURATION - MODIFY THESE PARAMETERS
    # =================================================
    
    # Base model configuration - select from ConfigPresets
    config_choice = 'debug'  # Options: 'debug', 'b1', 'b7', 'b14', 'b50', 'b100', 'b200', 'b300'
    
    # Override specific parameters (set to None to use preset defaults)
    override_params = {
        'use_moe': True,        # Enable MoE
        'num_epochs': 3,        # Training epochs (reduced for testing)
        'learning_rate': 1e-4,  # Learning rate
        'batch_size': 1,        # Micro batch size
        'gradient_accumulation_steps': 16,  # Reduced for testing
        'train_data_path': 'oasst1_data/oasst1_train.jsonl',
        'eval_data_path': 'oasst1_data/oasst1_train.jsonl',
        'capacity_factor': 1.25,
        'load_balancing_weight': 0.08,
    }
    
    # DeepSpeed and optimization settings - THESE ARE MANUAL OVERRIDES
    manual_deepspeed_overrides = {
        'use_deepspeed': DEEPSPEED_AVAILABLE,
        'cpu_offload': True,             # FORCE CPU offloading
        'cpu_offload_optimizer': True,   # FORCE CPU optimizer offloading
        'cpu_offload_parameters': True,  # FORCE CPU parameter offloading
        'zero_stage': 2,                 # FORCE ZeRO-2
        'nvme_path': "Deepspeed/Tmp",               # Set to NVMe path if available
    }
    
    # =================================================
    # END HARDCODED CONFIGURATION
    
    # Generate experiment name
    if args.continue_as:
        experiment_name = args.continue_as
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.resume:
            experiment_name = f'Resumed_DeepSpeed_MoE_{timestamp}'
        else:
            experiment_name = f'DeepSpeed_MoE_{override_params.get("num_experts", 8)}E_CPU_Offload_Fixed_{timestamp}'
    
    # Handle data processing commands
    if args.process_data:
        input_file, output_file = args.process_data
        print(f"Processing data: {input_file} -> {output_file}")
        try:
            result = process_oasst_data(input_file, output_file)
            if result:
                print("Data processing completed successfully")
            else:
                print("Data processing failed")
        except Exception as e:
            print(f"Data processing error: {e}")
        return
    
    if args.create_shards:
        train_data_path = Path(override_params['train_data_path'])
        create_data_shards(train_data_path, args.create_shards)
        return
    
    # Setup distributed environment first
    dist_manager = DeepSpeedIntegrationManager()
    dist_manager.setup_distributed_environment()
    
    # Only log from rank 0 in distributed training
    should_log = dist_manager.should_log()
    
    if should_log:
        # Create enhanced directory structure
        create_enhanced_directory_structure()
        
        # Setup enhanced logging
        setup_deepspeed_logging()
        
        print("\n" + "="*80)
        print("ENHANCED DEEPSPEED TRAINING SYSTEM - VERSION 2.0")
        print("="*80)
        print(f"Experiment: {experiment_name}")
        print(f"DeepSpeed Available: {DEEPSPEED_AVAILABLE}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        print("="*80)
        
        # System information
        resource_manager = EnhancedResourceManager(dist_manager)
        print("\nSystem Information:")
        for key, value in resource_manager.system_info.items():
            if key == 'gpu_info' and isinstance(value, list):
                print(f"  {key}:")
                for i, gpu in enumerate(value):
                    if 'error' in gpu:
                        print(f"    GPU {i}: Error - {gpu['error']}")
                    else:
                        print(f"    GPU {i}: {gpu.get('name', 'Unknown')} ({gpu.get('memory_gb', 0):.1f}GB)")
            elif key != 'distributed_info':
                print(f"  {key}: {value}")
        
        print("\nDistributed Information:")
        dist_info = resource_manager.system_info['distributed_info']
        for key, value in dist_info.items():
            print(f"  {key}: {value}")
    
    # Handle environment check
    if args.check_environment:
        if should_log:
            env_status = validate_environment()
            print(f"Environment validation: {env_status}")
        return
    
    # Handle debug setup
    if args.debug_setup:
        # Load base configuration for debugging
        if hasattr(ConfigPresets, config_choice):
            config = getattr(ConfigPresets, config_choice)()
        else:
            print(f"Unknown config preset: {config_choice}")
            return
        
        # Apply overrides
        for key, value in override_params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        debug_training_setup(config)
        return
    
    # Handle test deepspeed
    if args.test_deepspeed:
        # Load base configuration for testing
        if hasattr(ConfigPresets, config_choice):
            config = getattr(ConfigPresets, config_choice)()
        else:
            print(f"Unknown config preset: {config_choice}")
            return
        
        # Apply overrides
        for key, value in {**override_params, **manual_deepspeed_overrides}.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(config, key, value)
        
        test_deepspeed_setup(config)
        return
    
    try:
        # STEP 1: Create base configuration
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 1: CREATING BASE CONFIGURATION")
            print(f"{'='*60}")
        
        # Get base preset
        if hasattr(ConfigPresets, config_choice):
            config = getattr(ConfigPresets, config_choice)()
        else:
            raise ValueError(f"Unknown config preset: {config_choice}")
        
        # Handle resume functionality
        resume_state = None
        if args.resume and not args.force_restart:
            checkpoint_path = find_checkpoint_path(args.resume, experiment_name)
            if checkpoint_path:
                if should_log:
                    print(f"Found checkpoint for resume: {checkpoint_path}")
                
                # Analyze checkpoint if requested
                if args.analyze_model:
                    analyze_model_checkpoint(checkpoint_path)
                    return
                
                # For resume, we'll load the checkpoint after model creation
                # but we can extract some config info now
                try:
                    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                    if 'config' in checkpoint_data:
                        checkpoint_config = checkpoint_data['config']
                        if should_log:
                            print("Using configuration from checkpoint")
                        # Update config with checkpoint values for compatibility
                        for key, value in checkpoint_config.items():
                            if hasattr(config, key):
                                setattr(config, key, value)
                except Exception as e:
                    if should_log:
                        print(f"Could not load config from checkpoint: {e}")
                        print("Using default configuration")
            else:
                if should_log:
                    print(f"Checkpoint '{args.resume}' not found, starting fresh training")
                args.resume = None
        
        
        # Apply parameter overrides
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 2: APPLYING PARAMETER OVERRIDES")
            print(f"{'='*60}")
        
        for key, value in override_params.items():
            if value is not None:
                if hasattr(config, key):
                    old_value = getattr(config, key)
                    setattr(config, key, value)
                    if should_log:
                        print(f"OVERRIDE: {key}: {old_value} -> {value}")
                else:
                    setattr(config, key, value)
                    if should_log:
                        print(f"NEW PARAM: {key} = {value}")
        
        # STEP 3: Auto-configure DeepSpeed optimizations
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 3: AUTO-CONFIGURING DEEPSPEED OPTIMIZATIONS")
            print(f"{'='*60}")
        
        # Create DeepSpeed configuration wizard
        wizard = AdaptiveDeepSpeedWizard(resource_manager)
        
        # Apply DeepSpeed auto-configuration with manual overrides
        config = wizard.auto_configure_deepspeed(
            config, 
            natural_description="MoE training with CPU offloading for memory efficiency",
            manual_overrides=manual_deepspeed_overrides
        )
        
        # Validate configuration
        config_issues = validate_deepspeed_config(config)
        if config_issues and should_log:
            print(f"\n⚠️  Configuration Issues Detected:")
            for issue in config_issues:
                print(f"   • {issue}")
        
        # STEP 4: Create training components
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 4: CREATING TRAINING COMPONENTS")
            print(f"{'='*60}")
        
        # Create tokenizer
        if should_log:
            print("Creating tokenizer...")
        try:
            tokenizer = ConversationTokenizer()
            if should_log:
                print(f"   ✓ Tokenizer created: vocab_size={tokenizer.get_vocab_size()}")
        except Exception as e:
            print(f"   ✗ Tokenizer creation failed: {e}")
            return
        
        # Create model configuration
        if should_log:
            print("Creating model configuration...")
        try:
            model_config = config_to_deepseek_config(config)
            if should_log:
                print(f"   ✓ Model config: {model_config.hidden_size}d, {model_config.num_layers} layers")
                if model_config.use_moe:
                    print(f"      MoE enabled: {model_config.num_experts} experts, top-{model_config.moe_top_k}")
        except Exception as e:
            print(f"   ✗ Model config creation failed: {e}")
            traceback.print_exc()
            return
        
        # Create model
        if should_log:
            print("Creating model...")
        try:
            model = DeepSeekTransformer(model_config)
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            if should_log:
                print(f"   ✓ Model created successfully")
                print(f"      Total parameters: {param_count:,}")
                print(f"      Trainable parameters: {trainable_params:,}")
                
                # Memory footprint estimation
                memory_info = safe_model_memory_footprint(model)
                if 'total_size_mb' in memory_info:
                    print(f"      Estimated memory: {memory_info['total_size_mb']:.1f} MB")
        except Exception as e:
            print(f"   ✗ Model creation failed: {e}")
            traceback.print_exc()
            return
        
        # Handle benchmark performance
        if args.benchmark_performance:
            benchmark_training_performance(config)
            return
        
        # Handle memory analysis
        if args.memory_analysis:
            profile_memory_usage(config)
            return
        
        # Handle model analysis without checkpoint
        if args.analyze_model and not args.checkpoint:
            print("\nModel Analysis (Current Configuration):")
            print(f"  Architecture: {model_config.hidden_size}d, {model_config.num_layers} layers")
            print(f"  Parameters: {param_count:,}")
            print(f"  MoE: {'Yes' if model_config.use_moe else 'No'}")
            if model_config.use_moe:
                print(f"    Experts: {model_config.num_experts}")
                print(f"    Top-K: {model_config.moe_top_k}")
            return
        
        # Handle model export
        if args.export_model:
            if args.checkpoint:
                checkpoint_path = find_checkpoint_path(args.checkpoint, experiment_name)
                if checkpoint_path:
                    export_model_for_inference(checkpoint_path, Path(args.output_dir), args.format)
                else:
                    print(f"Checkpoint '{args.checkpoint}' not found for export")
            else:
                print("No checkpoint specified for export")
            return
        
        # STEP 5: Prepare training data
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 5: PREPARING TRAINING DATA")
            print(f"{'='*60}")
        
        # Check if training data exists, create dummy data if not
        train_data_path = Path(config.train_data_path)
        if not train_data_path.exists():
            if should_log:
                print(f"Training data not found at {train_data_path}")
                print("Creating dummy training data for testing...")
            create_dummy_training_data(train_data_path, num_samples=100)
        
        # Validate and report on data
        if args.validate_data or args.data_report:
            if should_log:
                print("Validating training data...")
            try:
                validation_result = validate_data_comprehensive(str(train_data_path))
                if should_log:
                    print(f"Data validation: {validation_result.get('status', 'unknown')}")
                
                if args.data_report:
                    report = create_data_summary_report(str(train_data_path))
                    print(f"Data summary report: {report}")
            except Exception as e:
                if should_log:
                    print(f"Data validation failed: {e}")
            
            if args.validate_data or args.data_report:
                return
        
        # Create datasets
        if should_log:
            print("Creating datasets...")
        try:
            # Use streaming dataset for large data
            train_dataset = StreamingConversationDataset(
                data_path=str(train_data_path),
                tokenizer=tokenizer,
                max_length=config.seq_length,
                conversation_format="oasst"
            )
            
            # Create eval dataset if path exists
            eval_dataset = None
            eval_data_path = Path(config.eval_data_path) if hasattr(config, 'eval_data_path') else train_data_path
            if eval_data_path.exists() and eval_data_path != train_data_path:
                eval_dataset = StreamingConversationDataset(
                    data_path=str(eval_data_path),
                    tokenizer=tokenizer,
                    max_length=config.seq_length,
                    conversation_format="oasst"
                )
            
            if should_log:
                print(f"   ✓ Training dataset created")
                if eval_dataset:
                    print(f"   ✓ Evaluation dataset created")
                else:
                    print(f"   • Using training data for evaluation")
        except Exception as e:
            print(f"   ✗ Dataset creation failed: {e}")
            traceback.print_exc()
            return
        
        # STEP 6: Initialize trainer
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 6: INITIALIZING TRAINER")
            print(f"{'='*60}")
        
        trainer = None
        if EnhancedConversationTrainer is not None:
            try:
                if should_log:
                    print("Creating enhanced trainer...")
                
                trainer = EnhancedConversationTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    logger=logging.getLogger(__name__),
                    experiment_name=experiment_name,
                    distributed_manager=dist_manager
                )
                
                if should_log:
                    print("   ✓ Enhanced trainer created successfully")
            except Exception as e:
                if should_log:
                    print(f"   ✗ Enhanced trainer creation failed: {e}")
                    print("   • Falling back to basic trainer...")
                trainer = None
        
        # Fallback to basic trainer if enhanced trainer failed
        if trainer is None:
            try:
                trainer = create_fallback_trainer(model, tokenizer, config)
                if should_log:
                    print("   ✓ Basic fallback trainer created")
            except Exception as e:
                print(f"   ✗ All trainer creation failed: {e}")
                traceback.print_exc()
                return
        
        # Handle resume state loading
        if args.resume and resume_state is None:
            checkpoint_path = find_checkpoint_path(args.resume, experiment_name)
            if checkpoint_path and hasattr(trainer, 'load_checkpoint'):
                try:
                    resume_state = trainer.load_checkpoint(checkpoint_path)
                    if should_log:
                        print(f"   ✓ Checkpoint loaded for resume")
                except Exception as e:
                    if should_log:
                        print(f"   ✗ Failed to load checkpoint: {e}")
        
        # STEP 7: Performance estimation
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 7: PERFORMANCE ESTIMATION")
            print(f"{'='*60}")
        
        if not args.dry_run:
            try:
                # Estimate training time
                perf_stats = benchmark_actual_performance(model, tokenizer, config, dist_manager, samples=5)
                
                if 'single_step_time' in perf_stats:
                    estimated_time = estimate_training_time(
                        config.num_epochs, 
                        len(train_dataset) if hasattr(train_dataset, '__len__') else 1000,
                        config.batch_size,
                        config.gradient_accumulation_steps,
                        perf_stats['single_step_time']
                    )
                    
                    if should_log:
                        print(f"   Estimated training time: {estimated_time.get('estimated_hours', 'unknown')} hours")
                        print(f"   Step time: {perf_stats['single_step_time']:.3f}s")
                        print(f"   Memory usage: {perf_stats.get('current_memory_gb', 0):.1f}GB")
            except Exception as e:
                if should_log:
                    print(f"   Performance estimation failed: {e}")
        
        # STEP 8: Start training or dry run
        if should_log:
            print(f"\n{'='*60}")
            print("STEP 8: STARTING TRAINING")
            print(f"{'='*60}")
        
        if args.dry_run:
            print("DRY RUN COMPLETED - Configuration validated successfully")
            print(f"Experiment: {experiment_name}")
            print("Ready to start actual training")
            return
        
        # Actual training
        try:
            if should_log:
                print(f"Starting training for experiment: {experiment_name}")
                print(f"Configuration: {config_choice} + overrides")
                print(f"DeepSpeed: {'Enabled' if getattr(config, 'use_deepspeed', False) else 'Disabled'}")
                print(f"MoE: {'Enabled' if getattr(config, 'use_moe', False) else 'Disabled'}")
            
            # Start training
            trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                resume_state=resume_state
            )
            
            if should_log:
                print(f"\n{'='*60}")
                print("TRAINING COMPLETED SUCCESSFULLY")
                print(f"{'='*60}")
                print(f"Experiment: {experiment_name}")
                print("Check the experiments directory for results and checkpoints")
        
        except KeyboardInterrupt:
            if should_log:
                print(f"\n{'='*60}")
                print("TRAINING INTERRUPTED BY USER")
                print(f"{'='*60}")
                print("Attempting to save emergency checkpoint...")
            
            # Try to save emergency checkpoint
            try:
                if hasattr(trainer, 'save_emergency_checkpoint'):
                    trainer.save_emergency_checkpoint()
                    print("Emergency checkpoint saved")
            except Exception as e:
                print(f"Failed to save emergency checkpoint: {e}")
        
        except Exception as e:
            if should_log:
                print(f"\n{'='*60}")
                print("TRAINING FAILED")
                print(f"{'='*60}")
                print(f"Error: {e}")
                traceback.print_exc()
            
            # Try to save emergency checkpoint
            try:
                if hasattr(trainer, 'save_emergency_checkpoint'):
                    trainer.save_emergency_checkpoint()
                    print("Emergency checkpoint saved")
            except:
                pass
        
        finally:
            # Cleanup
            cleanup_memory()
            
            if should_log:
                print(f"\n{'='*60}")
                print("CLEANUP COMPLETED")
                print(f"{'='*60}")
    
    except Exception as e:
        print(f"Critical error in main: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()