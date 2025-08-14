# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
try:
    from config.config_manager import Config, ConfigPresets
    from training.orchestrator import TrainingOrchestrator
    from utils.data_processing import process_oasst_data, validate_data_comprehensive
    from utils.environment import validate_environment, estimate_training_time
    from utils.reporting import create_data_summary_report
    from core.tokenizer import ConversationTokenizer
    from core.model import estimate_parameters
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are present in the correct directory structure.")
    sys.exit(1)


def setup_logging_basic():
    """Setup basic logging before full system initialization."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_directory_structure():
    """Create necessary directory structure."""
    directories = [
        'data',
        'checkpoints', 
        'experiments',
        'logs',
        'backups'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def main():
    """Enhanced main function with comprehensive CLI and error handling."""
    # Setup basic logging first
    setup_logging_basic()
    
    # Create directory structure
    create_directory_structure()
    
    parser = argparse.ArgumentParser(
        description='Production-Ready Conversational Transformer Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick debug run
  python Main.py --config debug --test-generation
  
  # Production training
  python Main.py --config medium --train-data data/train.jsonl --eval-data data/eval.jsonl
  
  # Resume training
  python Main.py --config medium --resume checkpoints/model_epoch_005.pt
  
  # Data processing and validation
  python Main.py --process-oasst raw_data.jsonl processed_data.jsonl
  python Main.py --validate-data processed_data.jsonl --create-report
  
  # Custom configuration
  python Main.py --config small --epochs 10 --lr 1e-4 --batch-size 8
        """
    )
    
    # Configuration options
    parser.add_argument('--config', choices=['debug', 'small', 'medium', 'large'], 
                       default='debug', help='Configuration preset')
    parser.add_argument('--config-file', type=str, help='Load config from YAML file')
    
    # Data options
    parser.add_argument('--train-data', type=str, default='data/train.jsonl',
                       help='Training data path')
    parser.add_argument('--eval-data', type=str, default='data/eval.jsonl',
                       help='Evaluation data path')
    
    # Training overrides
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--grad-accum', type=int, help='Override gradient accumulation steps')
    parser.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], help='Override precision')
    
    # Experiment options
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Testing and validation
    parser.add_argument('--test-generation', action='store_true', help='Test generation after training')
    parser.add_argument('--validate-data', type=str, help='Validate data file format')
    parser.add_argument('--create-report', action='store_true', help='Create comprehensive data report')
    
    # Data processing
    parser.add_argument('--process-oasst', nargs=2, metavar=('INPUT', 'OUTPUT'),
                       help='Process OASST data: input_file output_file')
    parser.add_argument('--max-conversations', type=int, help='Limit conversations processed')
    
    # System options
    parser.add_argument('--check-environment', action='store_true', help='Check training environment')
    parser.add_argument('--estimate-time', action='store_true', help='Estimate training time')
    parser.add_argument('--dry-run', action='store_true', help='Setup everything but don\'t train')
    
    args = parser.parse_args()
    
    # Environment validation
    if args.check_environment:
        logging.info("Checking training environment...")
        issues = validate_environment()
        if issues:
            logging.warning("Environment issues found:")
            for issue in issues:
                logging.warning(f"  - {issue}")
        else:
            logging.info("Environment looks good!")
        
        if not args.dry_run:
            return 0
    
    # Data processing
    if args.process_oasst:
        input_file, output_file = args.process_oasst
        try:
            count = process_oasst_data(input_file, output_file, args.max_conversations)
            logging.info(f"Successfully processed {count} conversations")
            return 0
        except Exception as e:
            logging.error(f"Data processing failed: {e}")
            return 1
    
    # Data validation
    if args.validate_data:
        try:
            tokenizer = ConversationTokenizer()
            stats = validate_data_comprehensive(args.validate_data, tokenizer)
            
            logging.info("Data Validation Results:")
            logging.info(f"  Valid conversations: {stats['conversation_stats']['valid_conversations']:,}")
            logging.info(f"  Success rate: {stats['quality_metrics']['success_rate']:.2%}")
            logging.info(f"  Average tokens: {stats['token_stats']['avg_tokens']:.1f}")
            
            if args.create_report:
                create_data_summary_report([args.validate_data], tokenizer)
            
            return 0
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            return 1
    
    # Load configuration
    try:
        if args.config_file:
            config = Config.load(args.config_file)
        else:
            config_map = {
                'debug': ConfigPresets.debug,
                'small': ConfigPresets.small,
                'medium': ConfigPresets.medium,
                'large': ConfigPresets.large,
            }
            config = config_map[args.config]()
        
        # Apply CLI overrides
        if args.epochs is not None:
            config.num_epochs = args.epochs
        if args.lr is not None:
            config.learning_rate = args.lr
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.grad_accum is not None:
            config.gradient_accumulation_steps = args.grad_accum
        if args.precision is not None:
            config.precision = args.precision
        if args.experiment_name is not None:
            config.experiment_name = args.experiment_name
        
        config.train_data_path = args.train_data
        config.eval_data_path = args.eval_data
        config.seed = args.seed
        
        # Re-validate after overrides
        config.validate()
        
    except Exception as e:
        logging.error(f"Configuration error: {e}")
        return 1
    
    # Training time estimation
    if args.estimate_time:
        try:
            # Estimate dataset size
            if Path(config.train_data_path).exists():
                with open(config.train_data_path, 'r') as f:
                    dataset_size = sum(1 for _ in f)
            else:
                dataset_size = 10000  # Default estimate
            
            estimates = estimate_training_time(config, dataset_size)
            
            logging.info("Training Time Estimates:")
            logging.info(f"  Dataset size: {dataset_size:,} conversations")
            logging.info(f"  Estimated time: {estimates['estimated_hours']:.1f} hours ({estimates['estimated_days']:.1f} days)")
            logging.info(f"  Total tokens: {estimates['total_tokens']:,}")
            logging.info(f"  Throughput: {estimates['tokens_per_second']:,} tokens/sec")
            logging.info(f"  Memory utilization: {estimates['memory_utilization']:.1%}")
            
            if estimates['memory_warning']:
                logging.warning("  ⚠️  High memory utilization expected - consider reducing batch size")
            
            if not args.dry_run:
                return 0
        except Exception as e:
            logging.error(f"Time estimation failed: {e}")
            return 1
    
    # Dry run
    if args.dry_run:
        logging.info("Dry run completed successfully!")
        return 0
    
    # Main training
    logging.info("="*80)
    logging.info("PRODUCTION CONVERSATIONAL TRANSFORMER TRAINING")
    logging.info("="*80)
    
    try:
        # Initialize training orchestrator
        orchestrator = TrainingOrchestrator(config)
        
        # Log configuration
        logging.info(f"Configuration: {args.config}")
        logging.info(f"Model parameters: ~{estimate_parameters(config):,}")
        logging.info(f"Experiment: {config.experiment_name}")
        
        # Run training
        orchestrator.run_training()
        
        # Test generation if requested
        if args.test_generation and orchestrator.trainer:
            logging.info("\n" + "="*60)
            logging.info("TESTING GENERATION")
            logging.info("="*60)
            
            test_prompts = [
                "Hello, how are you today?",
                "What is machine learning?",
                "Write a simple Python function to calculate factorial.",
                "Explain the concept of recursion in programming.",
                "What are the benefits of using transformers in NLP?"
            ]
            
            for i, prompt in enumerate(test_prompts, 1):
                logging.info(f"\nTest {i}/5:")
                logging.info(f"User: {prompt}")
                try:
                    response = orchestrator.trainer.generate(prompt)
                    logging.info(f"Assistant: {response}")
                except Exception as e:
                    logging.error(f"Generation failed: {e}")
                logging.info("-" * 50)
        
        logging.info("\n🎉 Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())