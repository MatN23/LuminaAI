# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import argparse
import traceback
import psutil
import gc
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
try:
    from config.config_manager import Config, ConfigPresets
    from training.orchestrator import AdaptiveTrainingOrchestrator, TrainingMetrics
    from utils.data_processing import process_oasst_data, validate_data_comprehensive
    from utils.environment import validate_environment, estimate_training_time
    from utils.reporting import create_data_summary_report
    from core.tokenizer import ConversationTokenizer
    from core.model import estimate_parameters, DeepSeekTransformer, DeepSeekConfig
    from core.dataset import ConversationDataset, StreamingConversationDataset, create_memory_efficient_dataloader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are present in the correct directory structure.")
    sys.exit(1)


class AdaptiveConfigurationWizard:
    """Natural language configuration system."""
    
    def __init__(self):
        self.config_templates = {
            'small_efficient': 'Optimized for small models (1-3B parameters) with maximum efficiency',
            'medium_balanced': 'Balanced configuration for medium models (7-14B parameters)',
            'large_performance': 'High-performance setup for large models (70B+ parameters)',
            'code_specialized': 'Specialized for code generation and programming tasks',
            'conversation_optimized': 'Optimized for conversational AI and chat applications',
            'research_experimental': 'Experimental setup for research and novel architectures'
        }
    
    def parse_natural_language_config(self, description: str) -> str:
        """Parse natural language description to determine config."""
        description_lower = description.lower()
        
        # Model size detection
        if any(word in description_lower for word in ['small', '1b', '3b', 'efficient', 'fast']):
            if 'code' in description_lower:
                return 'b3_inference'
            return 'debug'
        elif any(word in description_lower for word in ['medium', '7b', '14b', 'balanced']):
            return 'b14'
        elif any(word in description_lower for word in ['large', '70b', '100b', 'performance']):
            return 'b100'
        
        # Task-specific detection
        if any(word in description_lower for word in ['code', 'programming', 'developer']):
            return 'b6_quality'  # Good for code tasks
        elif any(word in description_lower for word in ['chat', 'conversation', 'assistant']):
            return 'b7'
        elif any(word in description_lower for word in ['research', 'experiment', 'novel']):
            return 'b50'
        
        # Default to balanced
        return 'b7'
    
    def get_config_explanation(self, config_choice: str) -> str:
        """Get human-readable explanation of config choice."""
        explanations = {
            'debug': 'Ultra-lightweight config for testing and debugging',
            'b1': '1B parameter model - very fast training and inference',
            'b7': '7B parameter model - balanced performance and efficiency', 
            'b14': '14B parameter model - high quality with reasonable resource usage',
            'b50': '50B parameter model - research-grade performance',
            'b100': '100B parameter model - maximum performance for production',
            'b200': '200B parameter model - cutting-edge large scale',
            'b300': '300B parameter model - experimental massive scale',
            'b3_inference': '3B model optimized for fast inference',
            'b6_quality': '6B model optimized for high-quality outputs',
            'm120_speed': '120M model optimized for maximum speed',
            'm70_memory': '70M model optimized for minimal memory usage'
        }
        return explanations.get(config_choice, f'Custom configuration: {config_choice}')


class IntelligentResourceManager:
    """Intelligent system resource management and optimization."""
    
    def __init__(self):
        self.system_info = self._gather_system_info()
        self.optimization_history = []
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information."""
        info = {
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 2000,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'memory_usage_percent': psutil.virtual_memory().percent
        }
        
        # GPU information
        try:
            import torch
            if torch.cuda.is_available():
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['cuda_version'] = torch.version.cuda
            else:
                info.update({'gpu_count': 0, 'gpu_memory_gb': 0, 'gpu_name': 'None'})
        except ImportError:
            info.update({'gpu_count': 0, 'gpu_memory_gb': 0, 'gpu_name': 'None'})
        
        # Storage information
        try:
            disk_usage = psutil.disk_usage('.')
            info['disk_free_gb'] = disk_usage.free / (1024**3)
            info['disk_total_gb'] = disk_usage.total / (1024**3)
        except Exception:
            info.update({'disk_free_gb': 100, 'disk_total_gb': 500})
        
        return info
    
    def intelligent_config_optimization(self, base_config, dataset_size: int) -> Dict[str, Any]:
        """Intelligently optimize configuration based on system resources and dataset."""
        optimizations = {}
        
        # Memory-based optimizations
        if self.system_info['memory_gb'] < 16:
            optimizations.update({
                'batch_size': min(base_config.batch_size, 2),
                'gradient_accumulation_steps': max(base_config.gradient_accumulation_steps, 8),
                'num_workers': min(2, self.system_info['cpu_count'] // 2),
                'max_shard_size_mb': 256,
                'enable_gradient_checkpointing': True,
                'precision': 'fp16',
                'reasoning': 'Low memory system - reduced batch size, increased accumulation'
            })
        elif self.system_info['memory_gb'] < 64:
            optimizations.update({
                'batch_size': min(base_config.batch_size, 8),
                'num_workers': min(4, self.system_info['cpu_count'] // 2),
                'max_shard_size_mb': 512,
                'precision': 'bf16' if self.system_info.get('gpu_memory_gb', 0) > 8 else 'fp16',
                'reasoning': 'Medium memory system - balanced optimization'
            })
        else:
            optimizations.update({
                'num_workers': min(8, self.system_info['cpu_count']),
                'max_shard_size_mb': 1024,
                'precision': 'bf16',
                'enable_flash_attention': True,
                'reasoning': 'High memory system - optimized for performance'
            })
        
        # GPU-based optimizations
        gpu_memory = self.system_info.get('gpu_memory_gb', 0)
        if gpu_memory > 0:
            if gpu_memory < 8:
                optimizations.update({
                    'batch_size': min(optimizations.get('batch_size', base_config.batch_size), 1),
                    'gradient_accumulation_steps': max(optimizations.get('gradient_accumulation_steps', base_config.gradient_accumulation_steps), 16),
                    'model_parallel': False,
                    'reasoning': optimizations.get('reasoning', '') + ' | Small GPU memory - aggressive batching'
                })
            elif gpu_memory > 24:
                optimizations.update({
                    'enable_model_parallel': True,
                    'enable_large_model_optimizations': True,
                    'reasoning': optimizations.get('reasoning', '') + ' | Large GPU memory - enabled advanced features'
                })
        
        # Dataset-based optimizations
        dataset_gb = dataset_size * 0.001  # Rough estimate
        if dataset_gb > 10:
            optimizations.update({
                'streaming_dataset': True,
                'prefetch_factor': 4,
                'persistent_workers': True,
                'reasoning': optimizations.get('reasoning', '') + ' | Large dataset - streaming optimizations'
            })
        elif dataset_gb < 1:
            optimizations.update({
                'streaming_dataset': False,
                'cache_dataset': True,
                'reasoning': optimizations.get('reasoning', '') + ' | Small dataset - in-memory caching'
            })
        
        # CPU optimizations
        if self.system_info['cpu_count'] > 16:
            optimizations.update({
                'enable_cpu_offload': True,
                'num_workers': min(12, self.system_info['cpu_count'] - 4),
                'reasoning': optimizations.get('reasoning', '') + ' | Many CPU cores - enabled CPU offload'
            })
        
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'system_info': self.system_info.copy(),
            'optimizations': optimizations.copy()
        })
        
        return optimizations
    
    def get_resource_recommendations(self) -> Dict[str, str]:
        """Get human-readable resource optimization recommendations."""
        recommendations = {}
        
        # Memory recommendations
        memory_usage = self.system_info['memory_usage_percent']
        if memory_usage > 85:
            recommendations['memory'] = 'WARNING: High memory usage. Consider closing other applications or reducing batch size.'
        elif memory_usage > 70:
            recommendations['memory'] = 'CAUTION: Moderate memory usage. Monitor during training.'
        else:
            recommendations['memory'] = 'OK: Memory usage is healthy.'
        
        # GPU recommendations
        gpu_memory = self.system_info.get('gpu_memory_gb', 0)
        if gpu_memory == 0:
            recommendations['gpu'] = 'INFO: No GPU detected. Training will use CPU (much slower).'
        elif gpu_memory < 8:
            recommendations['gpu'] = 'WARNING: Limited GPU memory. Use small models and aggressive optimizations.'
        elif gpu_memory < 24:
            recommendations['gpu'] = 'OK: Moderate GPU memory. Medium models should work well.'
        else:
            recommendations['gpu'] = 'EXCELLENT: High GPU memory. Large models supported.'
        
        # Storage recommendations
        free_space = self.system_info.get('disk_free_gb', 100)
        if free_space < 10:
            recommendations['storage'] = 'CRITICAL: Very low disk space. Training may fail.'
        elif free_space < 50:
            recommendations['storage'] = 'WARNING: Limited disk space. Monitor checkpoint storage.'
        else:
            recommendations['storage'] = 'OK: Sufficient disk space available.'
        
        return recommendations


class PredictiveTrainingAnalyzer:
    """Predicts training outcomes and provides intelligent suggestions."""
    
    def __init__(self):
        self.historical_data = []
    
    def predict_training_time(self, config, dataset_size: int, system_info: Dict) -> Dict[str, Any]:
        """Predict training time with multiple scenarios."""
        # Base calculations
        total_tokens = dataset_size * getattr(config, 'seq_length', 2048)
        batch_size = getattr(config, 'batch_size', 8)
        grad_accum = getattr(config, 'gradient_accumulation_steps', 1)
        effective_batch_size = batch_size * grad_accum
        
        # Model size factor
        model_size_factor = self._estimate_model_size_factor(config)
        
        # System performance factor
        gpu_memory = system_info.get('gpu_memory_gb', 0)
        if gpu_memory == 0:
            system_factor = 0.1  # CPU is much slower
        elif gpu_memory < 8:
            system_factor = 0.5
        elif gpu_memory < 24:
            system_factor = 1.0
        else:
            system_factor = 1.5
        
        # Base throughput estimation (tokens per second)
        base_throughput = 1000 * system_factor / model_size_factor
        
        # Training time scenarios
        scenarios = {
            'optimistic': {
                'throughput_multiplier': 1.5,
                'description': 'Everything goes perfectly, no issues'
            },
            'realistic': {
                'throughput_multiplier': 1.0,
                'description': 'Normal training with typical issues'
            },
            'pessimistic': {
                'throughput_multiplier': 0.6,
                'description': 'Training difficulties and optimization needed'
            }
        }
        
        predictions = {}
        for scenario, params in scenarios.items():
            actual_throughput = base_throughput * params['throughput_multiplier']
            total_training_time = total_tokens / actual_throughput
            
            predictions[scenario] = {
                'hours': total_training_time / 3600,
                'days': total_training_time / (3600 * 24),
                'tokens_per_second': actual_throughput,
                'description': params['description'],
                'total_tokens': total_tokens,
                'effective_batch_size': effective_batch_size
            }
        
        # Add recommendations
        predictions['recommendations'] = self._get_timing_recommendations(predictions, config)
        
        return predictions
    
    def _estimate_model_size_factor(self, config) -> float:
        """Estimate computational complexity factor based on model size."""
        hidden_size = getattr(config, 'hidden_size', 768)
        num_layers = getattr(config, 'num_layers', 12)
        
        # Rough parameter count estimation
        param_estimate = hidden_size * hidden_size * num_layers * 12  # Simplified
        
        if param_estimate < 1e9:  # < 1B parameters
            return 1.0
        elif param_estimate < 10e9:  # < 10B parameters
            return 3.0
        elif param_estimate < 100e9:  # < 100B parameters
            return 10.0
        else:  # >= 100B parameters
            return 30.0
    
    def _get_timing_recommendations(self, predictions: Dict, config) -> List[str]:
        """Get recommendations based on timing predictions."""
        recommendations = []
        
        realistic_hours = predictions['realistic']['hours']
        
        if realistic_hours > 72:  # More than 3 days
            recommendations.extend([
                "Consider using a smaller model for initial experiments",
                "Enable gradient checkpointing to reduce memory usage",
                "Use mixed precision training (fp16/bf16)",
                "Consider distributed training if multiple GPUs available"
            ])
        elif realistic_hours > 24:  # More than 1 day
            recommendations.extend([
                "Plan for overnight training sessions",
                "Enable automatic checkpointing every few hours",
                "Monitor training progress remotely"
            ])
        else:
            recommendations.append("Training time looks reasonable for experimentation")
        
        # Memory-based recommendations
        if predictions['realistic']['tokens_per_second'] < 100:
            recommendations.append("Low throughput detected - consider optimizing batch size or model architecture")
        
        return recommendations


class InteractiveTrainingDebugger:
    """Interactive debugging and analysis system."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.debug_history = []
    
    def analyze_current_state(self) -> Dict[str, Any]:
        """Provide comprehensive analysis of current training state."""
        status = self.orchestrator.get_adaptive_status()
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'unknown',
            'issues_detected': [],
            'recommendations': [],
            'metrics_summary': {},
            'adaptive_intelligence_status': {}
        }
        
        # Analyze current metrics
        if 'current_metrics' in status and status['current_metrics']:
            metrics = status['current_metrics']
            analysis['metrics_summary'] = {
                'loss': metrics.get('loss', 'unknown'),
                'learning_rate': metrics.get('learning_rate', 'unknown'),
                'gradient_norm': metrics.get('grad_norm', 'unknown'),
                'memory_usage': metrics.get('memory_usage', {}).get('gpu_memory_percent', 'unknown')
            }
            
            # Health assessment
            loss = metrics.get('loss', float('inf'))
            grad_norm = metrics.get('grad_norm', 0)
            
            if loss == float('inf') or loss > 10:
                analysis['overall_health'] = 'critical'
                analysis['issues_detected'].append('Loss is very high or infinite')
                analysis['recommendations'].append('Check data preprocessing and model initialization')
            elif grad_norm > 10:
                analysis['overall_health'] = 'warning'
                analysis['issues_detected'].append('High gradient norm detected')
                analysis['recommendations'].append('Consider gradient clipping or learning rate reduction')
            elif loss < 0.1:
                analysis['overall_health'] = 'excellent'
                analysis['recommendations'].append('Training is converging well')
            else:
                analysis['overall_health'] = 'good'
        
        # Analyze adaptive intelligence
        analysis['adaptive_intelligence_status'] = {
            'decisions_made': status.get('adaptive_decisions_made', 0),
            'metrics_collected': status.get('metrics_collected', 0),
            'meta_learning_runs': status.get('meta_learning_runs', 0),
            'monitoring_active': status.get('monitoring_active', False)
        }
        
        # Recent decisions analysis
        if 'recent_decisions' in status and status['recent_decisions']:
            decision_types = [d['type'] for d in status['recent_decisions']]
            confidence_avg = sum(d['confidence'] for d in status['recent_decisions']) / len(status['recent_decisions'])
            
            analysis['adaptive_intelligence_status'].update({
                'recent_decision_types': list(set(decision_types)),
                'average_confidence': confidence_avg,
                'most_recent_decision': status['recent_decisions'][-1]['type']
            })
            
            if confidence_avg < 0.5:
                analysis['issues_detected'].append('Low confidence in recent adaptive decisions')
                analysis['recommendations'].append('Consider manual intervention or configuration adjustment')
        
        self.debug_history.append(analysis)
        return analysis
    
    def get_debug_conversation(self) -> str:
        """Generate a natural language summary of training state."""
        analysis = self.analyze_current_state()
        
        conversation = f"""
Training Assistant Analysis ({datetime.now().strftime('%H:%M:%S')})

Overall Health: {analysis['overall_health'].upper()}

Current Metrics:
"""
        
        for metric, value in analysis['metrics_summary'].items():
            if isinstance(value, float):
                conversation += f"  • {metric.replace('_', ' ').title()}: {value:.4f}\n"
            else:
                conversation += f"  • {metric.replace('_', ' ').title()}: {value}\n"
        
        if analysis['issues_detected']:
            conversation += f"\nIssues Detected:\n"
            for issue in analysis['issues_detected']:
                conversation += f"  • {issue}\n"
        
        if analysis['recommendations']:
            conversation += f"\nRecommendations:\n"
            for rec in analysis['recommendations']:
                conversation += f"  • {rec}\n"
        
        ai_status = analysis['adaptive_intelligence_status']
        conversation += f"""
Adaptive AI Status:
  • Decisions Made: {ai_status['decisions_made']}
  • Metrics Collected: {ai_status['metrics_collected']}
  • Meta-learning Runs: {ai_status['meta_learning_runs']}
  • Monitoring: {'Active' if ai_status['monitoring_active'] else 'Inactive'}
"""
        
        if 'recent_decision_types' in ai_status:
            conversation += f"  • Recent Decision Types: {', '.join(ai_status['recent_decision_types'])}\n"
            conversation += f"  • Average Confidence: {ai_status['average_confidence']:.2f}\n"
        
        return conversation


def config_to_deepseek_config(config):
    """Convert training Config to DeepSeekConfig."""
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
        use_moe=getattr(config, 'use_moe', False),
        num_experts=getattr(config, 'num_experts', 8),
        moe_top_k=getattr(config, 'moe_top_k', 2),
        capacity_factor=getattr(config, 'capacity_factor', 1.25),
        load_balancing_weight=getattr(config, 'load_balancing_weight', 0.01),
    )


def setup_logging_advanced():
    """Setup advanced logging with real-time monitoring."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def create_enhanced_directory_structure():
    """Create comprehensive directory structure."""
    directories = [
        'data', 'data/shards', 'data/processed', 'data/cache',
        'checkpoints', 'checkpoints/best', 'checkpoints/emergency',
        'experiments', 'experiments/archive',
        'logs', 'logs/adaptive', 'logs/performance',
        'backups', 'backups/configs', 'backups/models',
        'reports', 'reports/adaptive', 'reports/performance',
        'monitoring', 'monitoring/metrics', 'monitoring/visualizations'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)


def interactive_configuration_setup():
    """Interactive setup for advanced users."""
    print("\n" + "="*70)
    print("ADAPTIVE AI TRAINING CONFIGURATION WIZARD")
    print("="*70)
    
    wizard = AdaptiveConfigurationWizard()
    
    print("\nAvailable configuration templates:")
    for key, description in wizard.config_templates.items():
        print(f"  • {key}: {description}")
    
    print(f"\nYou can also describe your needs in natural language!")
    print(f"Example: 'Train a 7B model optimized for code generation with aggressive memory optimization'")
    
    user_input = input("\nDescribe your training needs or choose a template: ").strip()
    
    if user_input.lower() in wizard.config_templates:
        config_choice = user_input.lower()
    else:
        config_choice = wizard.parse_natural_language_config(user_input)
    
    explanation = wizard.get_config_explanation(config_choice)
    print(f"\nSelected configuration: {config_choice}")
    print(f"Explanation: {explanation}")
    
    return config_choice


def main():
    """Enhanced main function with adaptive intelligence and natural language interface."""
    # Create enhanced directory structure first
    create_enhanced_directory_structure()
    
    # Setup advanced logging
    setup_logging_advanced()
    
    # Initialize intelligent components
    resource_manager = IntelligentResourceManager()
    
    print("\n" + "="*80)
    print("ADAPTIVE AI-DRIVEN TRANSFORMER TRAINING SYSTEM")
    print("   Self-Improving • Intelligent • Production-Ready")
    print("="*80)
    
    # Display system information
    print(f"\nSystem Information:")
    system_info = resource_manager.system_info
    print(f"   RAM: {system_info['memory_gb']:.1f}GB ({system_info['memory_usage_percent']:.1f}% used)")
    print(f"   CPU: {system_info['cpu_count']} cores @ {system_info['cpu_freq']:.0f}MHz")
    print(f"   GPU: {system_info['gpu_name']} ({system_info['gpu_memory_gb']:.1f}GB)" if system_info['gpu_memory_gb'] > 0 else "   GPU: None detected")
    print(f"   Storage: {system_info['disk_free_gb']:.1f}GB free")
    
    # Resource recommendations
    recommendations = resource_manager.get_resource_recommendations()
    print(f"\nSystem Assessment:")
    for component, recommendation in recommendations.items():
        status_emoji = "✅" if "OK" in recommendation or "EXCELLENT" in recommendation else "⚠️" if "WARNING" in recommendation or "CAUTION" in recommendation else "🚨"
        print(f"   {status_emoji} {component.upper()}: {recommendation}")
    
    # Configuration setup - you can modify these parameters
    config_choice = 'debug'  # Change this or use interactive_configuration_setup()
    natural_language_config = None  # e.g., "Train a 7B model optimized for code generation"
    
    # Uncomment for interactive setup:
    # config_choice = interactive_configuration_setup()
    
    # Parse natural language if provided
    if natural_language_config:
        wizard = AdaptiveConfigurationWizard()
        config_choice = wizard.parse_natural_language_config(natural_language_config)
        print(f"\nInterpreted '{natural_language_config}' as: {config_choice}")
    
    # Training parameters - modify as needed
    train_data = 'oasst1_data/oasst1_train.jsonl'
    eval_data = 'data/eval.jsonl'
    epochs = 10
    lr = 3e-4
    batch_size = 2
    grad_accum = 4
    precision = 'auto'
    inference_precision = 'auto'
    experiment_name = f'AdaptiveAI_Experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    seed = 42
    
    # Advanced features
    enable_adaptive_intelligence = True
    enable_meta_learning = True
    enable_real_time_monitoring = True
    enable_predictive_analytics = True
    enable_interactive_debugging = False  # Set to True for interactive mode
    
    # Validation and processing flags
    validate_data = None  # Set to data path to validate
    process_oasst = None  # Set to (input_file, output_file) to process
    create_report = False
    check_environment = False
    estimate_time = True
    dry_run = False
    
    # Environment validation
    if check_environment:
        logging.info("Validating training environment...")
        issues = validate_environment()
        if issues:
            logging.warning("Environment issues found:")
            for issue in issues:
                logging.warning(f"     - {issue}")
        else:
            logging.info("Environment validation passed!")
        
        if dry_run:
            return 0
    
    # Data processing
    if process_oasst:
        input_file, output_file = process_oasst
        try:
            count = process_oasst_data(input_file, output_file)
            logging.info(f"Successfully processed {count:,} conversations")
            return 0
        except Exception as e:
            logging.error(f"Data processing failed: {e}")
            return 1
    
    # Initialize tokenizer
    try:
        tokenizer = ConversationTokenizer(model_name="gpt-4")
        logging.info("Tokenizer initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize tokenizer: {e}")
        return 1
    
    # Data validation
    if validate_data:
        try:
            logging.info(f"Validating data: {validate_data}")
            
            validate_path = Path(validate_data)
            if not validate_path.exists():
                # Check for sharded files
                shard_files = list(validate_path.parent.glob(f"{validate_path.stem}_shard_*.jsonl"))
                if shard_files:
                    logging.info(f"Found {len(shard_files)} shard files")
                    validate_data = str(shard_files[0])
                else:
                    logging.error(f"File not found: {validate_data}")
                    return 1
            
            stats = validate_data_comprehensive(validate_data, tokenizer)
            
            if stats:
                logging.info("Validation Results:")
                # Log stats safely
                for key, value in stats.items():
                    if isinstance(value, dict):
                        continue
                    elif isinstance(value, float) and 0 <= value <= 1:
                        logging.info(f"   {key.replace('_', ' ').title()}: {value:.2%}")
                    elif isinstance(value, (int, float)):
                        logging.info(f"   {key.replace('_', ' ').title()}: {value:,}")
                
                if create_report:
                    create_data_summary_report([validate_data], tokenizer)
                    logging.info("Data summary report created")
            
            return 0
            
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            return 1
    
    # Load and optimize configuration
    try:
        config_map = {
            'debug': ConfigPresets.debug,
            'b1': ConfigPresets.b1,
            'b7': ConfigPresets.b7,
            'b14': ConfigPresets.b14,
            'b50': ConfigPresets.b50,
            'b100': ConfigPresets.b100,
            'b200': ConfigPresets.b200,
            'b300': ConfigPresets.b300,
            'b3_inference': ConfigPresets.b3_inference,
            'b6_quality': ConfigPresets.b6_quality,
            'm120_speed': ConfigPresets.m120_speed,
            'm70_memory': ConfigPresets.m70_memory
        }
        config = config_map[config_choice]()
        
        # Apply intelligent optimizations
        try:
            with open(train_data, 'r') as f:
                dataset_size = sum(1 for _ in f)
        except:
            dataset_size = 10000  # Default estimate
        
        optimizations = resource_manager.intelligent_config_optimization(config, dataset_size)
        
        # Apply optimizations to config
        for key, value in optimizations.items():
            if key != 'reasoning' and hasattr(config, key):
                setattr(config, key, value)
                logging.info(f"Optimization: {key} = {value}")
        
        if 'reasoning' in optimizations:
            logging.info(f"Optimization reasoning: {optimizations['reasoning']}")
        
        # Apply manual overrides
        config.num_epochs = epochs
        config.learning_rate = lr
        config.batch_size = batch_size
        config.gradient_accumulation_steps = grad_accum
        config.precision = precision
        config.inference_precision = inference_precision
        config.experiment_name = experiment_name
        config.train_data_path = train_data
        config.eval_data_path = eval_data
        config.seed = seed
        config.lr_scheduler = None
        
        config.validate()
        logging.info(f"Configuration loaded and optimized: {config_choice}")
        
    except Exception as e:
        logging.error(f"Configuration error: {e}")
        return 1
    
    # Predictive analysis
    if estimate_time:
        try:
            analyzer = PredictiveTrainingAnalyzer()
            predictions = analyzer.predict_training_time(config, dataset_size, system_info)
            
            print(f"\nTraining Time Predictions:")
            for scenario, pred in predictions.items():
                if scenario == 'recommendations':
                    continue
                print(f"   {scenario.upper()}: {pred['hours']:.1f} hours ({pred['days']:.1f} days)")
                print(f"     Throughput: {pred['tokens_per_second']:,.0f} tokens/sec")
                print(f"     {pred['description']}")
            
            if predictions['recommendations']:
                print(f"\nTiming Recommendations:")
                for rec in predictions['recommendations']:
                    print(f"   • {rec}")
            
        except Exception as e:
            logging.error(f"Time estimation failed: {e}")
    
    # Dry run
    if dry_run:
        print(f"\nDry Run Summary:")
        print(f"   Configuration: {config_choice}")
        print(f"   Dataset: {train_data}")
        print(f"   Estimated size: {dataset_size:,} conversations")
        print(f"   Training precision: {config.precision}")
        print(f"   Inference precision: {config.inference_precision}")
        print(f"   Adaptive features: {'Enabled' if enable_adaptive_intelligence else 'Disabled'}")
        print(f"\nDry run completed - ready for training!")
        return 0
    
    # Main adaptive training
    try:
        print(f"\nInitializing Adaptive Training System...")
        
        # Initialize adaptive orchestrator
        orchestrator = AdaptiveTrainingOrchestrator(config)
        
        # Create DeepSeek config for parameter estimation  
        deepseek_config = config_to_deepseek_config(config)
        
        # Display training information
        print(f"\nTraining Configuration:")
        print(f"   Experiment: {config.experiment_name}")
        print(f"   Model: ~{estimate_parameters(deepseek_config):,} parameters")
        print(f"   Dataset: {dataset_size:,} conversations")
        print(f"   Precision: {config.precision} (training) / {config.inference_precision} (inference)")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Batch size: {config.batch_size} (effective: {config.batch_size * config.gradient_accumulation_steps})")
        print(f"   Learning rate: {config.learning_rate}")
        
        # Adaptive features status
        print(f"\nAdaptive Intelligence Features:")
        print(f"   Meta-learning: {'Enabled' if enable_meta_learning else 'Disabled'}")
        print(f"   Real-time monitoring: {'Enabled' if enable_real_time_monitoring else 'Disabled'}")
        print(f"   Predictive analytics: {'Enabled' if enable_predictive_analytics else 'Disabled'}")
        print(f"   Interactive debugging: {'Enabled' if enable_interactive_debugging else 'Disabled'}")
        
        # Initialize interactive debugger if requested
        debugger = None
        if enable_interactive_debugging:
            debugger = InteractiveTrainingDebugger(orchestrator)
            print(f"\nInteractive debugging enabled - use debugger.get_debug_conversation() for insights")
        
        # Run adaptive training
        print(f"\n" + "="*60)
        print(f"STARTING ADAPTIVE TRAINING")
        print(f"="*60)
        
        start_time = datetime.now()
        orchestrator.run_adaptive_training()
        end_time = datetime.now()
        
        training_duration = (end_time - start_time).total_seconds()
        print(f"\nTraining completed successfully!")
        print(f"   Duration: {training_duration:.1f} seconds ({training_duration/3600:.2f} hours)")
        
        # Get final status
        final_status = orchestrator.get_adaptive_status()
        print(f"   Adaptive decisions made: {final_status.get('adaptive_decisions_made', 0)}")
        print(f"   Metrics collected: {final_status.get('metrics_collected', 0)}")
        print(f"   Meta-learning runs: {final_status.get('meta_learning_runs', 0)}")
        
        # Interactive debugging session
        if enable_interactive_debugging and debugger:
            print(f"\nFinal Training Analysis:")
            print(debugger.get_debug_conversation())
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        if 'orchestrator' in locals():
            orchestrator.cleanup()
        return 1
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(traceback.format_exc())
        if 'orchestrator' in locals():
            orchestrator.cleanup()
        return 1


if __name__ == "__main__":
    exit(main())