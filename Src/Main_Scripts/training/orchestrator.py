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
from typing import Dict, Any

import numpy as np
import torch

from config.config_manager import Config
from core.tokenizer import ConversationTokenizer
from core.model import TransformerModel
from core.dataset import ConversationDataset, create_dataloader
from monitoring.logger import ProductionLogger
from training.trainer import EnhancedConversationTrainer


class TrainingOrchestrator:
    """Orchestrates training with fault tolerance and monitoring."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = ProductionLogger(config.log_level, config.experiment_name)
        
        # Set random seeds for reproducibility
        self._set_seeds(config.seed)
        
        # Create experiment directory
        self.experiment_dir = Path(f"experiments/{config.experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config.save(str(self.experiment_dir / "config.yaml"))
        
        # Initialize components
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
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_training(self):
        """Initialize all training components."""
        logging.info("Initializing training components...")
        
        # Initialize tokenizer
        self.tokenizer = ConversationTokenizer()
        self.config.vocab_size = self.tokenizer.vocab_size
        
        # Initialize model
        self.model = TransformerModel(self.config)
        
        # Initialize trainer
        self.trainer = EnhancedConversationTrainer(
            self.model, self.tokenizer, self.config, self.logger
        )
        
        logging.info("Training components initialized successfully")
    
    def run_training(self):
        """Run the complete training pipeline with fault tolerance."""
        max_retries = self.config.max_retries
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
                    if self.config.auto_resume:
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
            self.trainer.train(train_dataset, eval_dataset)
            
            logging.info("Training completed successfully!")
            
        finally:
            self.is_training = False
            self.logger.close()
    
    def _setup_datasets(self):
        """Setup training and evaluation datasets."""
        logging.info("Setting up datasets...")
        
        # Validate data files
        if not Path(self.config.train_data_path).exists():
            raise FileNotFoundError(f"Training data not found: {self.config.train_data_path}")
        
        train_dataset = ConversationDataset(
            self.config.train_data_path, self.tokenizer, self.config, "train"
        )
        
        eval_dataset = None
        if Path(self.config.eval_data_path).exists():
            eval_dataset = ConversationDataset(
                self.config.eval_data_path, self.tokenizer, self.config, "eval"
            )
        
        return train_dataset, eval_dataset
    
    def _attempt_recovery(self):
        """Attempt to recover from the latest checkpoint."""
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return
        
        # Find latest checkpoint
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            return
        
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        
        try:
            logging.info(f"Attempting recovery from {latest_checkpoint}")
            if self.trainer:
                self.trainer.load_checkpoint(str(latest_checkpoint))
            logging.info("Recovery successful")
        except Exception as e:
            logging.error(f"Recovery failed: {e}")
    
    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint on failure."""
        if self.trainer and self.is_training:
            try:
                checkpoint_path = f"checkpoints/emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                self.trainer.save_checkpoint(self.trainer.current_epoch, emergency=True)
                logging.info(f"Emergency checkpoint saved: {checkpoint_path}")
            except Exception as e:
                logging.error(f"Failed to save emergency checkpoint: {e}")