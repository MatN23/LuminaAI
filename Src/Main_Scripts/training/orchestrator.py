# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import json
import logging
import math
import signal
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch

class TrainingOrchestrator:
    """Orchestrates training with fault tolerance and monitoring."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize logger
        try:
            from monitoring.logger import ProductionLogger
            self.logger = ProductionLogger(config.log_level, config.experiment_name)
        except ImportError:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility
        self._set_seeds(getattr(config, 'seed', 42))
        
        # Create experiment directory
        self.experiment_dir = Path(f"experiments/{config.experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        if hasattr(config, 'save'):
            config.save(str(self.experiment_dir / "config.yaml"))
        
        # Initialize components (delayed import to avoid circular dependencies)
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # State tracking
        self.is_training = False
        self.should_stop = False
        self.last_backup_time = time.time()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Health monitoring
        self.health_stats = {
            'last_loss': float('inf'),
            'loss_history': [],
            'consecutive_nan_batches': 0,
            'gradient_norm_history': []
        }
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # Ensure deterministic behavior
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_memory_management(self):
        """Setup memory management and optimization."""
        if torch.cuda.is_available():
            # Clear any existing memory
            torch.cuda.empty_cache()
            
            # Set memory fraction to prevent OOM
            memory_fraction = getattr(self.config, 'gpu_memory_fraction', 0.85)
            try:
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                logging.info(f"Set GPU memory fraction to {memory_fraction}")
            except Exception as e:
                logging.warning(f"Could not set memory fraction: {e}")
            
            # Enable memory mapping for large models
            try:
                import os
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                logging.info("Enabled expandable segments for CUDA memory")
            except Exception as e:
                logging.warning(f"Could not set CUDA alloc conf: {e}")
            
            # Log memory status
            self._log_gpu_memory_status()
    
    def _log_gpu_memory_status(self):
        """Log current GPU memory status."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                free = total_memory - allocated
                
                logging.info(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, "
                           f"{free:.2f}GB free of {total_memory:.2f}GB total")
    
    def _optimize_memory_usage(self):
        """Optimize memory usage before training."""
        # Clear unnecessary cached memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Set memory efficient attention if available
        if hasattr(self.model, 'enable_memory_efficient_attention'):
            self.model.enable_memory_efficient_attention()
            logging.info("Enabled memory efficient attention")
        
        # Enable gradient checkpointing if available
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing")
        
        # Adjust batch size based on available memory
        self._adjust_batch_size_for_memory()
    
    def _adjust_batch_size_for_memory(self):
        """Automatically adjust batch size based on available GPU memory."""
        if not torch.cuda.is_available():
            return
            
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            current_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            available_memory = total_memory - current_allocated
            
            # Conservative memory usage - leave 2GB buffer
            usable_memory = max(0, available_memory - 2.0)
            
            # Rough estimate: reduce batch size if memory is tight
            if usable_memory < 4.0:  # Less than 4GB available
                original_batch_size = getattr(self.config, 'batch_size', 8)
                if usable_memory < 2.0:
                    new_batch_size = max(1, original_batch_size // 4)
                elif usable_memory < 3.0:
                    new_batch_size = max(1, original_batch_size // 2)
                else:
                    new_batch_size = max(1, int(original_batch_size * 0.75))
                
                if new_batch_size != original_batch_size:
                    self.config.batch_size = new_batch_size
                    logging.warning(f"Reduced batch size from {original_batch_size} to {new_batch_size} "
                                  f"due to limited GPU memory ({usable_memory:.1f}GB available)")
                    
                    # Also adjust accumulation steps to maintain effective batch size
                    if hasattr(self.config, 'gradient_accumulation_steps'):
                        accumulation_factor = original_batch_size // new_batch_size
                        self.config.gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1) * accumulation_factor
                        logging.info(f"Increased gradient accumulation steps to {self.config.gradient_accumulation_steps}")
                        
        except Exception as e:
            logging.warning(f"Could not adjust batch size automatically: {e}")
    
    def _handle_oom_error(self, error):
        """Handle Out of Memory errors with comprehensive recovery strategies."""
        logging.error(f"CUDA Out of Memory Error: {error}")
        
        recovery_strategies = [
            ("Clear CUDA cache", self._clear_cuda_cache),
            ("Disable torch.compile", self._disable_torch_compile),
            ("Reduce batch size", self._reduce_batch_size),
            ("Enable gradient checkpointing", self._enable_gradient_checkpointing),
            ("Switch to CPU fallback", self._switch_to_cpu_fallback)
        ]
        
        for strategy_name, strategy_func in recovery_strategies:
            try:
                logging.info(f"Attempting recovery strategy: {strategy_name}")
                if strategy_func():
                    logging.info(f"Recovery strategy '{strategy_name}' successful")
                    # Clear cache after each successful strategy
                    self._clear_cuda_cache()
                    return True
            except Exception as e:
                logging.error(f"Recovery strategy '{strategy_name}' failed: {e}")
                continue
        
        logging.error("All OOM recovery strategies failed")
        return False
    
    def _clear_cuda_cache(self):
        """Clear CUDA cache and optimize memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logging.info("Cleared CUDA cache")
            self._log_gpu_memory_status()
            return True
        return False
    
    def _reduce_batch_size(self):
        """Reduce batch size as recovery strategy."""
        current_batch_size = getattr(self.config, 'batch_size', 1)
        if current_batch_size > 1:
            new_batch_size = max(1, current_batch_size // 2)
            self.config.batch_size = new_batch_size
            logging.info(f"Reduced batch size from {current_batch_size} to {new_batch_size}")
            
            # Adjust gradient accumulation to maintain effective batch size
            if hasattr(self.config, 'gradient_accumulation_steps'):
                self.config.gradient_accumulation_steps *= 2
                logging.info(f"Increased gradient accumulation to {self.config.gradient_accumulation_steps}")
            
            return True
        return False
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        if self.model and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing for memory savings")
            return True
        return False
    
    def _switch_to_cpu_fallback(self):
        """Switch to CPU as last resort."""
        if self.model:
            device = next(self.model.parameters()).device
            if device.type == 'cuda':
                self.model = self.model.cpu()
                torch.cuda.empty_cache()
                logging.warning("Switched model to CPU due to memory constraints")
                return True
        return False
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.should_stop = True
            if self.trainer and hasattr(self.trainer, 'should_stop'):
                self.trainer.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_training(self):
        """Initialize all training components."""
        logging.info("Initializing training components...")
        
        try:
            # Initialize tokenizer
            from core.tokenizer import ConversationTokenizer
            self.tokenizer = ConversationTokenizer()
            if hasattr(self.config, 'vocab_size'):
                self.config.vocab_size = self.tokenizer.vocab_size
            
            # Initialize model
            from core.model import DeepSeekTransformer
            
            # Try to get DeepSeek config
            model_config = self.config
            try:
                # Try to import config converter if it exists
                from Main import config_to_deepseek_config
                model_config = config_to_deepseek_config(self.config)
                logging.info("Using converted DeepSeek config")
            except ImportError:
                logging.info("Using config directly for model initialization")
            
            self.model = DeepSeekTransformer(model_config)
            logging.info("Model initialized successfully")
            
            # Initialize trainer (delayed import to avoid circular dependency)
            self._initialize_trainer()
            
            logging.info("All training components initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize training components: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def _initialize_trainer(self):
        """Initialize trainer with error handling for imports."""
        trainer_classes = [
            ('training.trainer', 'EnhancedConversationTrainer'),
            ('trainer', 'EnhancedConversationTrainer'),
            ('training.trainer', 'ConversationTrainer'),
            ('trainer', 'ConversationTrainer'),
        ]
        
        for module_name, class_name in trainer_classes:
            try:
                module = __import__(module_name, fromlist=[class_name])
                trainer_class = getattr(module, class_name)
                self.trainer = trainer_class(
                    self.model, self.tokenizer, self.config, self.logger
                )
                logging.info(f"Trainer initialized: {class_name} from {module_name}")
                return
            except (ImportError, AttributeError) as e:
                logging.debug(f"Could not import {class_name} from {module_name}: {e}")
                continue
        
        # Fallback: create a basic trainer
        logging.warning("Could not import any trainer class, creating basic trainer")
        self.trainer = self._create_basic_trainer()
    
    def _create_basic_trainer(self):
        """Create a basic trainer as fallback."""
        class BasicTrainer:
            def __init__(self, model, tokenizer, config, logger):
                self.model = model
                self.tokenizer = tokenizer
                self.config = config
                self.logger = logger
                self.current_epoch = 0
                self.global_step = 0
                self.best_eval_loss = float('inf')
                self.patience_counter = 0
                self.should_stop = False
            
            def train(self, train_dataset, eval_dataset=None):
                logging.info("Using basic trainer - implement your own trainer for full functionality")
                # Basic training loop would go here
                pass
            
            def save_checkpoint(self, step, emergency=False):
                checkpoint_path = f"checkpoint_step_{step}.pt"
                if emergency:
                    checkpoint_path = f"emergency_{checkpoint_path}"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'step': step,
                    'epoch': self.current_epoch
                }, checkpoint_path)
                logging.info(f"Checkpoint saved: {checkpoint_path}")
            
            def load_checkpoint(self, path):
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.global_step = checkpoint.get('step', 0)
                self.current_epoch = checkpoint.get('epoch', 0)
                return self.current_epoch
        
        return BasicTrainer(self.model, self.tokenizer, self.config, self.logger)
    
    def run_training(self):
        """Run the complete training pipeline with fault tolerance."""
        max_retries = getattr(self.config, 'max_retries', 3)
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                self._run_training_iteration()
                break  # Success
                
            except KeyboardInterrupt:
                logging.info("Training interrupted by user")
                self._save_emergency_checkpoint()
                break
                
            except Exception as e:
                retry_count += 1
                logging.error(f"Training failed (attempt {retry_count}/{max_retries + 1}): {e}")
                logging.error(traceback.format_exc())
                
                if retry_count <= max_retries:
                    logging.info(f"Retrying training in 30 seconds...")
                    time.sleep(30)
                    
                    # Try to recover from last checkpoint
                    if getattr(self.config, 'auto_resume', True):
                        self._attempt_recovery()
                else:
                    logging.error("Maximum retries exceeded, training failed")
                    self._save_emergency_checkpoint()
                    raise
    
    def _run_training_iteration(self):
        """Single training iteration."""
        self.is_training = True
        
        try:
            # Initialize if not already done
            if self.trainer is None:
                self.initialize_training()
            
            # Setup datasets
            train_dataset, eval_dataset = self._setup_datasets()
            
            # Run training
            if hasattr(self.trainer, 'train'):
                self.trainer.train(train_dataset, eval_dataset)
            else:
                logging.error("Trainer does not have a train method")
                raise AttributeError("Trainer missing train method")
            
            logging.info("Training completed successfully!")
            
        finally:
            self.is_training = False
            if hasattr(self.logger, 'close'):
                self.logger.close()
    
    def _setup_datasets(self):
        """Setup training and evaluation datasets."""
        logging.info("Setting up datasets...")
        
        # Validate data files
        train_data_path = Path(self.config.train_data_path)
        if not train_data_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.config.train_data_path}")
        
        # Import dataset class
        try:
            from core.dataset import ConversationDataset
        except ImportError:
            try:
                from dataset import ConversationDataset
            except ImportError:
                raise ImportError("Could not import ConversationDataset")
        
        # Create training dataset
        train_dataset = ConversationDataset(
            str(train_data_path), self.tokenizer, self.config, "train"
        )
        
        # Create evaluation dataset if specified
        eval_dataset = None
        if (hasattr(self.config, 'eval_data_path') and 
            self.config.eval_data_path and 
            Path(self.config.eval_data_path).exists()):
            eval_dataset = ConversationDataset(
                self.config.eval_data_path, self.tokenizer, self.config, "eval"
            )
        
        logging.info(f"Train dataset: {len(train_dataset)} samples")
        if eval_dataset:
            logging.info(f"Eval dataset: {len(eval_dataset)} samples")
        
        return train_dataset, eval_dataset
    
    def _attempt_recovery(self):
        """Attempt to recover from the latest checkpoint."""
        try:
            if self.trainer and hasattr(self.trainer, 'checkpoint_manager'):
                resume_path = self.trainer.checkpoint_manager.get_resume_path()
                if resume_path:
                    logging.info(f"Attempting recovery from {resume_path}")
                    epoch = self.trainer.load_checkpoint(resume_path)
                    logging.info(f"Recovery successful - resumed from epoch {epoch}")
                    return True
        except Exception as e:
            logging.error(f"Recovery from checkpoint manager failed: {e}")
        
        # Fallback: try to find any recent checkpoint
        checkpoint_dirs = [
            Path("checkpoints"),
            Path(f"experiments/{self.config.experiment_name}"),
            self.experiment_dir,
            Path(".")
        ]
        
        for checkpoint_dir in checkpoint_dirs:
            if not checkpoint_dir.exists():
                continue
                
            # Find latest checkpoint files
            checkpoint_patterns = ["*.pt", "*checkpoint*.pth", "model_*.bin"]
            checkpoints = []
            
            for pattern in checkpoint_patterns:
                checkpoints.extend(list(checkpoint_dir.glob(pattern)))
            
            if not checkpoints:
                continue
            
            # Sort by modification time and try the most recent
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            
            try:
                logging.info(f"Attempting recovery from {latest_checkpoint}")
                if self.trainer and hasattr(self.trainer, 'load_checkpoint'):
                    epoch = self.trainer.load_checkpoint(str(latest_checkpoint))
                    logging.info(f"Recovery successful from {latest_checkpoint} - epoch {epoch}")
                    return True
                else:
                    # Try to load checkpoint manually
                    checkpoint = torch.load(str(latest_checkpoint), map_location='cpu')
                    if 'model_state_dict' in checkpoint and self.model:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        logging.info(f"Model state loaded from {latest_checkpoint}")
                        return True
            except Exception as e:
                logging.error(f"Recovery from {latest_checkpoint} failed: {e}")
                continue
        
        logging.warning("No valid checkpoints found for recovery")
        return False
    
    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint on failure."""
        if not self.is_training:
            return
            
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            emergency_path = self.experiment_dir / f"emergency_checkpoint_{timestamp}.pt"
            
            if self.trainer and hasattr(self.trainer, 'save_checkpoint'):
                self.trainer.save_checkpoint(
                    getattr(self.trainer, 'global_step', 0), 
                    emergency=True
                )
                logging.info(f"Emergency checkpoint saved via trainer")
            elif self.model:
                # Manual emergency save
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'timestamp': timestamp,
                    'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
                }
                torch.save(checkpoint, emergency_path)
                logging.info(f"Emergency checkpoint saved manually: {emergency_path}")
        except Exception as e:
            logging.error(f"Failed to save emergency checkpoint: {e}")
    
    def validate_configuration(self):
        """Validate training configuration."""
        required_attrs = [
            'experiment_name', 'train_data_path', 'num_epochs',
            'learning_rate', 'batch_size', 'seq_length'
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            logging.error(f"Missing required config attributes: {missing_attrs}")
            return False
        
        # Validate paths
        if not Path(self.config.train_data_path).exists():
            logging.error(f"Training data path does not exist: {self.config.train_data_path}")
            return False
        
        # Validate numeric values
        numeric_checks = [
            ('learning_rate', lambda x: x > 0, "Learning rate must be positive"),
            ('batch_size', lambda x: x > 0, "Batch size must be positive"),
            ('num_epochs', lambda x: x > 0, "Number of epochs must be positive"),
            ('seq_length', lambda x: x > 0, "Sequence length must be positive")
        ]
        
        for attr, check, message in numeric_checks:
            value = getattr(self.config, attr)
            if not check(value):
                logging.error(f"{message}: got {value}")
                return False
        
        logging.info("Configuration validation passed")
        return True
    
    def get_training_status(self):
        """Get current training status."""
        status = {
            'is_training': self.is_training,
            'should_stop': self.should_stop,
            'experiment_name': self.config.experiment_name,
            'experiment_dir': str(self.experiment_dir),
        }
        
        if self.trainer:
            trainer_status = {}
            for attr in ['current_epoch', 'global_step', 'best_eval_loss', 'patience_counter']:
                trainer_status[attr] = getattr(self.trainer, attr, None)
            status.update(trainer_status)
        
        return status
    
    def stop_training(self):
        """Signal training to stop gracefully."""
        logging.info("Graceful training stop requested")
        self.should_stop = True
        if self.trainer and hasattr(self.trainer, 'should_stop'):
            self.trainer.should_stop = True
    
    def cleanup(self):
        """Clean up resources."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logging.info("CUDA cache cleared")
            except Exception as e:
                logging.error(f"Failed to clear CUDA cache: {e}")
        
        if hasattr(self.logger, 'close'):
            try:
                self.logger.close()
                logging.info("Logger closed")
            except Exception as e:
                logging.error(f"Failed to close logger: {e}")
        
        logging.info("Training orchestrator cleanup completed")


def create_training_orchestrator(config):
    """Factory function to create a TrainingOrchestrator instance."""
    return TrainingOrchestrator(config)


def main():
    """Main function for running the orchestrator standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Training Orchestrator')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate configuration, do not train')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        from config.config_manager import Config
        config = Config.load(args.config)
    except ImportError:
        logging.error("Could not import Config class")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create orchestrator
    orchestrator = TrainingOrchestrator(config)
    
    # Validate configuration
    if not orchestrator.validate_configuration():
        logging.error("Configuration validation failed")
        sys.exit(1)
    
    if args.validate_only:
        logging.info("Configuration validation passed - exiting")
        sys.exit(0)
    
    # Run training
    try:
        orchestrator.run_training()
        logging.info("Training completed successfully")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    main()