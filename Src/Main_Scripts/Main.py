# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import traceback
import psutil
import gc
import json
import time
import math
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# DeepSpeed imports
try:
    import deepspeed
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
    from core.tokenizer import ConversationTokenizer
    from core.model import DeepSeekTransformer, DeepSeekConfig
    from core.dataset import ConversationDataset, create_dataloader
except ImportError:
    try:
        from tokenizer import ConversationTokenizer
        from model import DeepSeekTransformer, DeepSeekConfig
        from dataset import ConversationDataset, create_dataloader
    except ImportError:
        print("ERROR: Could not import core modules")
        sys.exit(1)

# Try to import advanced training infrastructure
TRAINING_INFRASTRUCTURE_AVAILABLE = False
try:
    from training.orchestrator import AdaptiveTrainingOrchestrator
    from training.trainer import EnhancedConversationTrainer
    from training.checkpoint import CheckpointManager
    TRAINING_INFRASTRUCTURE_AVAILABLE = True
    print("Advanced training infrastructure available")
except ImportError:
    try:
        from orchestrator import AdaptiveTrainingOrchestrator
        from trainer import EnhancedConversationTrainer
        from checkpoint import CheckpointManager
        TRAINING_INFRASTRUCTURE_AVAILABLE = True
        print("Advanced training infrastructure available")
    except ImportError:
        print("Advanced training infrastructure not available - using built-in trainer")


def validate_precision_support(precision: str, device: torch.device) -> Tuple[bool, str]:
    """
    Validate if the requested precision is supported by the hardware.
    
    Returns:
        (is_supported, error_message)
    """
    if precision in ['fp32', 'float32']:
        return True, ""
    
    if device.type == 'cpu':
        if precision in ['fp16', 'mixed_fp16']:
            return False, f"FP16 precision '{precision}' is not supported on CPU. Use 'fp32' or 'bf16' instead."
        elif precision in ['bf16', 'mixed_bf16']:
            # Check if CPU supports bfloat16
            try:
                test_tensor = torch.randn(2, 2, dtype=torch.bfloat16)
                _ = test_tensor + test_tensor
                return True, ""
            except:
                return False, f"BF16 precision '{precision}' is not supported on this CPU. Use 'fp32' instead."
    
    if device.type == 'cuda':
        # Get GPU compute capability
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability
        
        if precision in ['fp16', 'mixed_fp16']:
            # FP16 supported on compute capability >= 5.3
            if major > 5 or (major == 5 and minor >= 3):
                return True, ""
            else:
                return False, f"FP16 precision '{precision}' requires compute capability >= 5.3. " \
                             f"Your GPU has {major}.{minor}. Use 'fp32' instead."
        
        elif precision in ['bf16', 'mixed_bf16']:
            # BF16 requires compute capability >= 8.0 (Ampere and newer)
            if major >= 8:
                return True, ""
            else:
                return False, f"BF16 precision '{precision}' requires compute capability >= 8.0 (Ampere GPU or newer). " \
                             f"Your GPU has {major}.{minor}. Use 'fp16' or 'fp32' instead."
    
    return True, ""


class ImprovedTimeEstimator:
    """Enhanced training time estimator with warmup handling and exponential smoothing."""
    
    def __init__(self, total_epochs: int, steps_per_epoch: int, warmup_steps: int = 10):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.warmup_steps = warmup_steps
        
        # Timing data
        self.step_times: deque = deque(maxlen=100)  # Keep last 100 steps
        self.start_time: Optional[float] = None
        self.last_step_time: Optional[float] = None
        self.epoch_start_times: Dict[int, float] = {}
        
        # Statistics tracking
        self.completed_steps = 0
        self.warmup_complete = False
        
        # Exponential smoothing parameters
        self.alpha = 0.3  # Smoothing factor (0-1, higher = more weight on recent data)
        self.smoothed_step_time: Optional[float] = None
        
        # Performance tracking
        self.min_step_time = float('inf')
        self.max_step_time = 0.0
        
    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
        self.last_step_time = self.start_time
        print(f"[TimeEstimator] Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[TimeEstimator] Warmup period: {self.warmup_steps} steps")
        print(f"[TimeEstimator] Total steps to complete: {self.total_steps:,}")
    
    def start_epoch(self, epoch: int):
        """Mark the start of an epoch."""
        self.epoch_start_times[epoch] = time.time()
    
    def record_step(self, step_num: int):
        """Record the completion of a training step."""
        current_time = time.time()
        
        if self.last_step_time is not None:
            step_duration = current_time - self.last_step_time
            
            # Filter out extreme outliers (likely due to checkpointing or evaluation)
            if len(self.step_times) > 0:
                median = sorted(self.step_times)[len(self.step_times) // 2]
                # Ignore steps that are 5x slower than median (likely checkpoint/eval)
                if step_duration > median * 5:
                    self.last_step_time = current_time
                    return
            
            self.step_times.append(step_duration)
            self.completed_steps += 1
            
            # Update min/max
            self.min_step_time = min(self.min_step_time, step_duration)
            self.max_step_time = max(self.max_step_time, step_duration)
            
            # Exponential smoothing
            if self.smoothed_step_time is None:
                self.smoothed_step_time = step_duration
            else:
                self.smoothed_step_time = (
                    self.alpha * step_duration + 
                    (1 - self.alpha) * self.smoothed_step_time
                )
            
            # Check if warmup is complete
            if not self.warmup_complete and self.completed_steps >= self.warmup_steps:
                self.warmup_complete = True
                print(f"\n[TimeEstimator] Warmup complete after {self.warmup_steps} steps")
                print(f"[TimeEstimator] Average warmup step time: {self.get_average_step_time():.3f}s")
        
        self.last_step_time = current_time
    
    def get_average_step_time(self) -> float:
        """Get average time per step (using all data)."""
        if not self.step_times:
            return 0.0
        return sum(self.step_times) / len(self.step_times)
    
    def get_recent_step_time(self, window: int = 20) -> float:
        """Get average of recent steps."""
        if not self.step_times:
            return 0.0
        recent = list(self.step_times)[-window:]
        return sum(recent) / len(recent)
    
    def get_smoothed_step_time(self) -> float:
        """Get exponentially smoothed step time (most reliable for prediction)."""
        if self.smoothed_step_time is None:
            return self.get_average_step_time()
        return self.smoothed_step_time
    
    def get_step_time_std(self) -> float:
        """Get standard deviation of step times."""
        if len(self.step_times) < 2:
            return 0.0
        avg = self.get_average_step_time()
        variance = sum((t - avg) ** 2 for t in self.step_times) / len(self.step_times)
        return math.sqrt(variance)
    
    def estimate_remaining_time(self, current_step: int) -> Dict[str, Any]:
        """
        Estimate remaining training time with confidence intervals.
        
        Returns:
            Dictionary with time estimates and statistics
        """
        if not self.step_times or current_step == 0:
            return {
                'status': 'initializing',
                'remaining_steps': self.total_steps,
                'estimated_remaining_seconds': 0.0,
                'eta': 'Calculating...',
                'progress_percent': 0.0,
                'warmup_complete': False
            }
        
        # Use smoothed time for most accurate prediction
        predicted_step_time = self.get_smoothed_step_time()
        
        # If still in warmup, use average instead
        if not self.warmup_complete:
            predicted_step_time = self.get_average_step_time()
        
        remaining_steps = self.total_steps - current_step
        estimated_remaining_seconds = remaining_steps * predicted_step_time
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        estimated_total_seconds = elapsed_time + estimated_remaining_seconds
        
        # Calculate ETA
        eta_datetime = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
        
        # Progress calculation
        progress_percent = (current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        # Calculate confidence interval (±1 std dev)
        std_dev = self.get_step_time_std()
        confidence_range_seconds = remaining_steps * std_dev
        
        eta_min = eta_datetime - timedelta(seconds=confidence_range_seconds)
        eta_max = eta_datetime + timedelta(seconds=confidence_range_seconds)
        
        # Calculate throughput metrics
        steps_per_second = 1.0 / predicted_step_time if predicted_step_time > 0 else 0.0
        
        return {
            'status': 'warmup' if not self.warmup_complete else 'training',
            
            # Step timing
            'avg_step_time': self.get_average_step_time(),
            'recent_step_time': self.get_recent_step_time(),
            'smoothed_step_time': predicted_step_time,
            'min_step_time': self.min_step_time if self.min_step_time != float('inf') else 0.0,
            'max_step_time': self.max_step_time,
            'step_time_std': std_dev,
            
            # Progress
            'current_step': current_step,
            'remaining_steps': remaining_steps,
            'total_steps': self.total_steps,
            'progress_percent': progress_percent,
            
            # Time estimates
            'elapsed_seconds': elapsed_time,
            'estimated_remaining_seconds': estimated_remaining_seconds,
            'estimated_total_seconds': estimated_total_seconds,
            
            # ETAs
            'eta': eta_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'eta_min': eta_min.strftime('%Y-%m-%d %H:%M:%S'),
            'eta_max': eta_max.strftime('%Y-%m-%d %H:%M:%S'),
            
            # Throughput
            'steps_per_second': steps_per_second,
            'steps_per_minute': steps_per_second * 60,
            'steps_per_hour': steps_per_second * 3600,
            
            # Status
            'warmup_complete': self.warmup_complete,
            'samples_collected': len(self.step_times),
        }
    
    def format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            hours = (seconds % 86400) / 3600
            return f"{days:.0f}d {hours:.1f}h"
    
    def format_datetime_relative(self, dt: datetime) -> str:
        """Format datetime relative to now."""
        now = datetime.now()
        if dt < now:
            return "Now"
        
        delta = dt - now
        seconds = delta.total_seconds()
        
        if seconds < 60:
            return "< 1 minute"
        elif seconds < 3600:
            return f"in {seconds/60:.0f} minutes"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"in {hours:.1f} hours"
        else:
            return dt.strftime('%b %d at %H:%M')
    
    def print_estimate(self, current_step: int, show_detailed: bool = False):
        """Print formatted time estimate."""
        estimate = self.estimate_remaining_time(current_step)
        
        if estimate['status'] == 'initializing':
            print("  [Time] Collecting timing data...")
            return
        
        # Format times
        elapsed = self.format_time(estimate['elapsed_seconds'])
        remaining = self.format_time(estimate['estimated_remaining_seconds'])
        total = self.format_time(estimate['estimated_total_seconds'])
        
        # Status indicator
        status_text = "WARMUP" if not estimate['warmup_complete'] else "TRAINING"
        
        # Main progress line
        print(f"\n  [{status_text}] Progress: {estimate['progress_percent']:.1f}% "
              f"({estimate['current_step']:,}/{estimate['total_steps']:,} steps)")
        
        # Time breakdown
        print(f"         Elapsed: {elapsed} | Remaining: ~{remaining} | Total: ~{total}")
        
        # ETA with confidence interval
        eta_relative = self.format_datetime_relative(
            datetime.strptime(estimate['eta'], '%Y-%m-%d %H:%M:%S')
        )
        print(f"         ETA: {estimate['eta']} ({eta_relative})")
        
        if estimate['warmup_complete'] and estimate['step_time_std'] > 0:
            confidence_margin = self.format_time(estimate['step_time_std'] * estimate['remaining_steps'])
            print(f"         Confidence: +/- {confidence_margin}")
        
        # Performance metrics
        print(f"         Speed: {estimate['steps_per_second']:.2f} steps/s | "
              f"{estimate['steps_per_minute']:.0f} steps/min | "
              f"Avg: {estimate['smoothed_step_time']:.3f}s/step")
        
        # Detailed statistics (optional)
        if show_detailed:
            print(f"         Step timing: "
                  f"min={estimate['min_step_time']:.3f}s, "
                  f"max={estimate['max_step_time']:.3f}s, "
                  f"std={estimate['step_time_std']:.3f}s")
            print(f"         Samples: {estimate['samples_collected']} | "
                  f"Status: {estimate['status']}")
    
    def get_epoch_summary(self, epoch: int) -> Dict[str, Any]:
        """Get summary for completed epoch."""
        if epoch not in self.epoch_start_times:
            return {}
        
        epoch_time = time.time() - self.epoch_start_times[epoch]
        
        return {
            'epoch': epoch,
            'epoch_time': epoch_time,
            'epoch_time_formatted': self.format_time(epoch_time),
            'steps_completed': self.completed_steps,
            'avg_step_time': self.get_average_step_time(),
        }
    
    def print_epoch_summary(self, epoch: int, total_epochs: int):
        """Print epoch completion summary."""
        summary = self.get_epoch_summary(epoch)
        if not summary:
            return
        
        remaining_epochs = total_epochs - epoch
        estimated_remaining = remaining_epochs * summary['epoch_time']
        
        print(f"\n  Epoch {epoch}/{total_epochs} completed in {summary['epoch_time_formatted']}")
        print(f"    Remaining epochs: {remaining_epochs} (~{self.format_time(estimated_remaining)})")


def create_dummy_training_data(file_path: Path, num_samples: int = 100):
    """Create dummy training data for testing."""
    print(f"Creating dummy training data: {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    dummy_conversations = []
    for i in range(num_samples):
        conversation = {
            "messages": [
                {"role": "user", "content": f"This is test question number {i+1}. Can you help me with a simple math problem? What is {i+2} + {i+3}?"},
                {"role": "assistant", "content": f"Of course! {i+2} + {i+3} = {(i+2) + (i+3)}. This is test response {i+1} for training purposes."}
            ]
        }
        dummy_conversations.append(conversation)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for conv in dummy_conversations:
            f.write(json.dumps(conv) + '\n')
    
    print(f"Created {num_samples} dummy conversations in {file_path}")


def config_to_deepseek_config(config: Config):
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
        
        # MoE configuration
        use_moe=getattr(config, 'use_moe', False),
        num_experts=getattr(config, 'num_experts', 8),
        moe_top_k=getattr(config, 'moe_top_k', 2),
        capacity_factor=getattr(config, 'capacity_factor', 1.25),
        load_balancing_weight=getattr(config, 'load_balancing_weight', 0.01),
    )


class ProductionTrainer:
    """Complete production trainer with DeepSpeed, MoE, and proper training loop."""
    
    def __init__(self, model, tokenizer, config, logger=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        # Time estimation
        self.time_estimator = None
        
        # Metrics tracking
        self.metrics = {
            'train_losses': [],
            'eval_losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'throughput': [],
            'epoch_times': []
        }
        
        # DeepSpeed integration
        self.use_deepspeed = DEEPSPEED_AVAILABLE and getattr(config, 'use_deepspeed', False)
        self.deepspeed_engine = None
        
        # Setup training components
        self._setup_training()
    
    def _setup_training(self):
        """Setup training components based on DeepSpeed availability."""
        if self.use_deepspeed:
            self._setup_deepspeed_training()
        else:
            self._setup_standard_training()
    
    def _setup_deepspeed_training(self):
        """Setup DeepSpeed training."""
        print("Setting up DeepSpeed training...")
        
        ds_config = self._create_deepspeed_config()
        
        try:
            self.deepspeed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=self.model,
                config=ds_config,
                model_parameters=self.model.parameters()
            )
            
            self.optimizer = optimizer
            self.scheduler = lr_scheduler
            self.model = self.deepspeed_engine
            
            print("DeepSpeed initialization successful!")
            
        except Exception as e:
            print(f"DeepSpeed initialization failed: {e}")
            print("Falling back to standard training...")
            self.use_deepspeed = False
            self._setup_standard_training()
    
    def _setup_standard_training(self):
        """Setup standard PyTorch training."""
        print("Setting up standard PyTorch training...")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_standard_optimizer()
        self.scheduler = None
        
        # Mixed precision setup
        self.training_precision = getattr(self.config, 'precision', 'fp32')
        self.use_amp = self.training_precision in ["fp16", "bf16", "mixed_fp16", "mixed_bf16"] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp and self.training_precision in ["fp16", "mixed_fp16"] else None
        
        # Model compilation
        if getattr(self.config, 'compile', False) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='default')
                print("Model compiled successfully")
            except Exception as e:
                print(f"Model compilation failed: {e}")
    
    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Create DeepSpeed configuration."""
        micro_batch_size = getattr(self.config, 'batch_size', 1)
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        train_batch_size = micro_batch_size * gradient_accumulation_steps * world_size
        
        ds_config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.learning_rate,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": getattr(self.config, 'weight_decay', 0.01)
                }
            },
            
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": getattr(self.config, 'min_lr', 1e-6),
                    "warmup_max_lr": self.config.learning_rate,
                    "warmup_num_steps": 1000
                }
            },
            
            "gradient_clipping": getattr(self.config, 'max_grad_norm', 1.0),
            "steps_per_print": 1,
            "wall_clock_breakdown": False
        }
        
        # Precision settings
        precision = getattr(self.config, 'precision', 'fp32')
        if precision in ["fp16", "mixed_fp16"]:
            ds_config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "auto_cast": False,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "consecutive_hysteresis": False
            }
        elif precision in ["bf16", "mixed_bf16"]:
            ds_config["bf16"] = {
                "enabled": True
            }
        
        # ZeRO configuration
        zero_stage = getattr(self.config, 'zero_stage', 2)
        if zero_stage > 0:
            zero_config = {
                "stage": zero_stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1000000000,
                "reduce_bucket_size": 500000000,
                "allgather_partitions": True,
                "reduce_scatter": True,
                "allgather_bucket_size": 500000000,
            }
            
            if getattr(self.config, 'cpu_offload', False):
                zero_config["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
                
            if getattr(self.config, 'cpu_offload_parameters', False):
                zero_config["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
            
            ds_config["zero_optimization"] = zero_config
        
        return ds_config
    
    def _create_standard_optimizer(self) -> torch.optim.Optimizer:
        """Create standard PyTorch optimizer."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'norm', 'embed']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': getattr(self.config, 'weight_decay', 0.01)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
    
    def _get_autocast_context(self):
        """Get autocast context based on precision."""
        if self.use_deepspeed:
            return torch.no_grad()  # DeepSpeed handles precision
        
        if not self.use_amp:
            return torch.no_grad()
        
        precision = self.training_precision
        if precision in ["fp16", "mixed_fp16"]:
            return autocast('cuda', dtype=torch.float16)
        elif precision in ["bf16", "mixed_bf16"]:
            return autocast('cuda', dtype=torch.bfloat16)
        else:
            return torch.no_grad()
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute loss for next-token prediction."""
        # For next-token prediction, shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten tensors
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # Create attention mask (ignore padding tokens)
        mask = (flat_labels != 0).float()  # Assuming 0 is pad token
        
        # Compute loss
        loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        masked_loss = loss * mask
        
        # Average over valid tokens
        total_loss = masked_loss.sum()
        total_weight = mask.sum().clamp(min=1)
        final_loss = total_loss / total_weight
        
        # Compute additional metrics
        with torch.no_grad():
            predictions = torch.argmax(flat_logits, dim=-1)
            correct_predictions = (predictions == flat_labels).float() * mask
            accuracy = correct_predictions.sum() / total_weight
            
            raw_loss = (loss * mask).sum() / total_weight
            perplexity = torch.exp(torch.clamp(raw_loss, min=0.0, max=10.0))
        
        return {
            'loss': final_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
            'valid_tokens': total_weight
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step."""
        if self.use_deepspeed:
            return self._deepspeed_train_step(batch)
        else:
            return self._standard_train_step(batch)
    
    def _deepspeed_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """DeepSpeed training step with proper MoE aux_loss handling."""
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        input_ids = batch.get('input_ids')
        labels = batch.get('labels', input_ids)
        
        if input_ids is None or input_ids.numel() == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf')}
        
        try:
            # Forward pass
            outputs = self.deepspeed_engine(input_ids)
            
            # Handle different output formats from MoE model
            logits = None
            aux_loss = None
            
            if isinstance(outputs, tuple):
                if len(outputs) >= 1:
                    logits = outputs[0]
                if len(outputs) >= 3:
                    potential_aux = outputs[2]
                    
                    if potential_aux is not None:
                        if isinstance(potential_aux, torch.Tensor):
                            if potential_aux.numel() > 0 and not torch.isnan(potential_aux).any():
                                aux_loss = potential_aux
                        elif isinstance(potential_aux, list):
                            valid_losses = []
                            for loss_item in potential_aux:
                                if isinstance(loss_item, torch.Tensor):
                                    if loss_item.numel() > 0 and not torch.isnan(loss_item).any():
                                        valid_losses.append(loss_item)
                            
                            if valid_losses:
                                aux_loss = sum(valid_losses)
            else:
                logits = outputs
            
            if logits is None:
                print(f"Error: No logits found in model output: {type(outputs)}")
                return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf')}
            
            if not isinstance(logits, torch.Tensor):
                print(f"Error: Logits is not a tensor: {type(logits)}")
                return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf')}
            
            # Compute loss
            loss_dict = self.compute_loss(logits, labels)
            main_loss = loss_dict['loss']
            
            # Add auxiliary loss if present
            total_loss = main_loss
            if aux_loss is not None:
                total_loss = main_loss + aux_loss
            
            # Validate loss before backward pass
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Invalid loss detected: {total_loss}")
                return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf')}
            
            # Backward pass
            self.deepspeed_engine.backward(total_loss)
            
            return {
                'loss': total_loss.item(),
                'accuracy': loss_dict['accuracy'].item(),
                'perplexity': loss_dict['perplexity'].item()
            }
            
        except Exception as e:
            print(f"DeepSpeed training step error: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf')}
    
    def _standard_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Standard PyTorch training step with proper MoE aux_loss handling."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        input_ids = batch.get('input_ids')
        labels = batch.get('labels', input_ids)
        
        if input_ids is None or input_ids.numel() == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf')}
        
        # Forward pass with precision
        with self._get_autocast_context():
            outputs = self.model(input_ids)
            
            logits = None
            aux_loss = None
            
            if isinstance(outputs, tuple):
                if len(outputs) >= 1:
                    logits = outputs[0]
                if len(outputs) >= 3:
                    potential_aux = outputs[2]
                    
                    if potential_aux is not None:
                        if isinstance(potential_aux, torch.Tensor):
                            if potential_aux.numel() > 0 and not torch.isnan(potential_aux).any():
                                aux_loss = potential_aux
                        elif isinstance(potential_aux, list):
                            valid_losses = []
                            for loss_item in potential_aux:
                                if isinstance(loss_item, torch.Tensor):
                                    if loss_item.numel() > 0 and not torch.isnan(loss_item).any():
                                        valid_losses.append(loss_item)
                            
                            if valid_losses:
                                aux_loss = sum(valid_losses)
            else:
                logits = outputs
            
            if logits is None:
                print(f"Error: No logits found in model output: {type(outputs)}")
                return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf')}
            
            if not isinstance(logits, torch.Tensor):
                print(f"Error: Logits is not a tensor: {type(logits)}")
                return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf')}
            
            # Compute main loss
            loss_dict = self.compute_loss(logits, labels)
            main_loss = loss_dict['loss']
            
            # Add auxiliary loss if present
            total_loss = main_loss
            if aux_loss is not None:
                total_loss = main_loss + aux_loss
        
        # Validate final loss before backward pass
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Invalid final loss detected: {total_loss}")
            return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf')}
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        return {
            'loss': total_loss.item(),
            'accuracy': loss_dict['accuracy'].item(),
            'perplexity': loss_dict['perplexity'].item()
        }
    
    def optimizer_step(self) -> Dict[str, float]:
        """Optimizer step."""
        if self.use_deepspeed:
            return self._deepspeed_optimizer_step()
        else:
            return self._standard_optimizer_step()
    
    def _deepspeed_optimizer_step(self) -> Dict[str, float]:
        """DeepSpeed optimizer step."""
        self.deepspeed_engine.step()
        
        # Get learning rate
        current_lr = self.config.learning_rate
        try:
            if hasattr(self.deepspeed_engine, 'get_lr'):
                lr_list = self.deepspeed_engine.get_lr()
                if lr_list and len(lr_list) > 0:
                    current_lr = lr_list[0]
        except:
            pass
        
        # Get gradient norm
        grad_norm = 0.0
        try:
            if hasattr(self.deepspeed_engine, 'get_global_grad_norm'):
                norm = self.deepspeed_engine.get_global_grad_norm()
                if norm is not None and not (math.isnan(norm) or math.isinf(norm)):
                    grad_norm = float(norm)
        except:
            pass
        
        return {'grad_norm': grad_norm, 'lr': current_lr}
    
    def _standard_optimizer_step(self) -> Dict[str, float]:
        """Standard optimizer step."""
        # Unscale gradients for AMP
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Compute gradient norm before clipping
        max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # Optimizer step
        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Get current learning rate
        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
        
        return {'grad_norm': grad_norm.item(), 'lr': current_lr}
    
    @torch.no_grad()
    def evaluate(self, eval_dataset, max_batches: int = 50) -> Dict[str, float]:
        """Evaluation."""
        if self.use_deepspeed:
            self.deepspeed_engine.eval()
        else:
            self.model.eval()
        
        eval_dataloader = create_dataloader(eval_dataset, self.config, shuffle=False)
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= max_batches:
                break
            
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            input_ids = batch.get('input_ids')
            labels = batch.get('labels', input_ids)
            
            if input_ids is None or input_ids.numel() == 0:
                continue
            
            try:
                # Forward pass
                if self.use_deepspeed:
                    outputs = self.deepspeed_engine(input_ids)
                else:
                    with self._get_autocast_context():
                        outputs = self.model(input_ids)
                
                # Handle outputs
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss_dict = self.compute_loss(logits, labels)
                
                if not (torch.isnan(loss_dict['loss']) or torch.isinf(loss_dict['loss'])):
                    total_loss += loss_dict['loss'].item()
                    total_accuracy += loss_dict['accuracy'].item()
                    total_perplexity += loss_dict['perplexity'].item()
                    num_batches += 1
                    
            except Exception as e:
                print(f"Evaluation error on batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            return {
                'eval_loss': float('inf'),
                'eval_accuracy': 0.0,
                'eval_perplexity': float('inf')
            }
        
        return {
            'eval_loss': total_loss / num_batches,
            'eval_accuracy': total_accuracy / num_batches,
            'eval_perplexity': total_perplexity / num_batches
        }
    
    def train_epoch(self, train_dataloader, epoch: int):
        """Train one epoch."""
        if self.use_deepspeed:
            self.deepspeed_engine.train()
        else:
            self.model.train()
        
        # Start epoch timing
        if self.time_estimator:
            self.time_estimator.start_epoch(epoch)
        
        epoch_metrics = {
            'total_loss': 0.0,
            'total_accuracy': 0.0,
            'total_perplexity': 0.0,
            'num_batches': 0,
            'grad_norm_sum': 0.0
        }
        
        accumulation_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'perplexity': 0.0
        }
        
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        print(f"Starting epoch {epoch + 1} with {len(train_dataloader)} batches")
        
        for batch_idx, batch in enumerate(train_dataloader):
            if self.should_stop:
                break
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Skip invalid batches
            if step_metrics['loss'] == 0.0 or math.isnan(step_metrics['loss']):
                print(f"Skipping batch {batch_idx} due to invalid loss")
                continue
            
            # Accumulate metrics
            accumulation_metrics['loss'] += step_metrics['loss'] / gradient_accumulation_steps
            accumulation_metrics['accuracy'] += step_metrics['accuracy']
            accumulation_metrics['perplexity'] += step_metrics['perplexity']
            
            # Optimizer step after accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                opt_metrics = self.optimizer_step()
                self.global_step += 1
                
                # Record step time
                if self.time_estimator:
                    self.time_estimator.record_step(self.global_step)
                
                # Update epoch metrics
                epoch_metrics['total_loss'] += accumulation_metrics['loss']
                epoch_metrics['total_accuracy'] += accumulation_metrics['accuracy'] / gradient_accumulation_steps
                epoch_metrics['total_perplexity'] += accumulation_metrics['perplexity'] / gradient_accumulation_steps
                epoch_metrics['num_batches'] += 1
                epoch_metrics['grad_norm_sum'] += opt_metrics['grad_norm']
                
                # Log progress
                if self.global_step % 10 == 0:
                    print(f"  Step {self.global_step:4d} | "
                          f"Loss: {accumulation_metrics['loss']:.4f} | "
                          f"Acc: {accumulation_metrics['accuracy']/gradient_accumulation_steps:.3f} | "
                          f"PPL: {accumulation_metrics['perplexity']/gradient_accumulation_steps:.2f} | "
                          f"LR: {opt_metrics['lr']:.6f} | "
                          f"GradNorm: {opt_metrics['grad_norm']:.4f}")
                    
                    # Print time estimate (every 10 steps after warmup)
                    if self.time_estimator and self.global_step > 5:
                        # Show detailed stats every 50 steps
                        show_detailed = (self.global_step % 50 == 0)
                        self.time_estimator.print_estimate(self.global_step, show_detailed=show_detailed)
                
                # Reset accumulation metrics
                accumulation_metrics = {'loss': 0.0, 'accuracy': 0.0, 'perplexity': 0.0}
        
        # Compute epoch statistics
        if epoch_metrics['num_batches'] > 0:
            avg_loss = epoch_metrics['total_loss'] / epoch_metrics['num_batches']
            avg_accuracy = epoch_metrics['total_accuracy'] / epoch_metrics['num_batches']
            avg_perplexity = epoch_metrics['total_perplexity'] / epoch_metrics['num_batches']
            avg_grad_norm = epoch_metrics['grad_norm_sum'] / epoch_metrics['num_batches']
        else:
            avg_loss = avg_accuracy = avg_perplexity = avg_grad_norm = 0.0
        
        # Print epoch summary
        if self.time_estimator:
            self.time_estimator.print_epoch_summary(epoch + 1, self.config.num_epochs)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_loss:.6f}")
        print(f"  Avg Accuracy: {avg_accuracy:.3f}")
        print(f"  Avg Perplexity: {avg_perplexity:.2f}")
        print(f"  Avg Grad Norm: {avg_grad_norm:.4f}")
        
        return {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'avg_perplexity': avg_perplexity,
            'avg_grad_norm': avg_grad_norm,
        }
    
    def train(self, train_dataset, eval_dataset=None):
        """Main training loop."""
        print("="*80)
        print("STARTING PRODUCTION TRAINING")
        print("="*80)
        
        # Setup data loaders
        train_dataloader = create_dataloader(train_dataset, self.config, shuffle=True)
        
        if len(train_dataloader) == 0:
            print("ERROR: Train dataloader is empty!")
            return
        
        # Initialize time estimator
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
        warmup_steps = min(10, steps_per_epoch // 10)  # Use 10% of epoch or 10 steps
        self.time_estimator = ImprovedTimeEstimator(self.config.num_epochs, steps_per_epoch, warmup_steps)
        self.time_estimator.start_training()
        
        print(f"\nTraining Configuration:")
        print(f"  Total epochs: {self.config.num_epochs}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total training steps: {self.time_estimator.total_steps}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.batch_size * gradient_accumulation_steps}")
        
        # Setup scheduler for standard training
        if not self.use_deepspeed and self.config.lr_scheduler == "cosine":
            total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=total_steps, 
                eta_min=getattr(self.config, 'min_lr', 1e-6)
            )
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                if self.should_stop:
                    break
                
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
                print(f"{'='*60}")
                
                # Train epoch
                epoch_metrics = self.train_epoch(train_dataloader, epoch)
                
                # Evaluation
                if eval_dataset is not None:
                    print("\nRunning evaluation...")
                    eval_metrics = self.evaluate(eval_dataset)
                    epoch_metrics.update(eval_metrics)
                    
                    print(f"\nEvaluation Results:")
                    print(f"  Eval Loss: {eval_metrics['eval_loss']:.6f}")
                    print(f"  Eval Accuracy: {eval_metrics['eval_accuracy']:.3f}")
                    print(f"  Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                    
                    # Early stopping check
                    if hasattr(self.config, 'early_stopping_patience') and self.config.early_stopping_patience:
                        self._check_early_stopping(eval_metrics['eval_loss'])
                
                # Save checkpoint
                self._save_checkpoint(epoch + 1)
                
                self.current_epoch = epoch + 1
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        except Exception as e:
            print(f"\n\nTraining error: {e}")
            traceback.print_exc()
            raise
        finally:
            total_training_time = time.time() - training_start_time
            
            print("\n" + "="*80)
            print("TRAINING COMPLETED")
            print("="*80)
            print(f"Total training time: {self.time_estimator.format_time(total_training_time)}")
            print(f"Total steps completed: {self.global_step}")
            print(f"Epochs completed: {self.current_epoch}/{self.config.num_epochs}")
            
            if self.time_estimator and len(self.time_estimator.step_times) > 0:
                avg_step = self.time_estimator.get_average_step_time()
                print(f"Average step time: {avg_step:.3f}s ({1.0/avg_step:.2f} steps/s)")
            
            # Final checkpoint
            self._save_checkpoint(self.current_epoch, final=True)
    
    def _check_early_stopping(self, eval_loss: float):
        """Check early stopping condition."""
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            print(f"  New best eval loss: {self.best_eval_loss:.6f}")
        else:
            self.patience_counter += 1
            print(f"  No improvement. Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
            
        if self.patience_counter >= self.config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {self.patience_counter} epochs without improvement")
            self.should_stop = True
    
    def _save_checkpoint(self, epoch: int, final: bool = False):
        """Save checkpoint."""
        try:
            suffix = "final" if final else f"epoch_{epoch:03d}"
            checkpoint_dir = Path(f"checkpoints/{suffix}_step_{self.global_step}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            if self.use_deepspeed:
                # DeepSpeed checkpoint
                self.deepspeed_engine.save_checkpoint(str(checkpoint_dir))
                print(f"DeepSpeed checkpoint saved: {checkpoint_dir}")
            else:
                # Standard PyTorch checkpoint
                checkpoint_path = checkpoint_dir / "pytorch_model.pt"
                checkpoint_data = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'global_step': self.global_step,
                    'epoch': epoch,
                    'config': self.config,
                    'best_eval_loss': self.best_eval_loss
                }
                torch.save(checkpoint_data, checkpoint_path)
                print(f"PyTorch checkpoint saved: {checkpoint_path}")
                
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")


def main():
    """Main training function."""
    
    # CONFIGURATION SECTION - MODIFY THESE PARAMETERS
    # =================================================
    
    # Base model configuration
    config_choice = 'debug'  # Options: 'debug', 'debug_200m', 'b1', 'b7', 'b14', 'b50', 'b100', 'b200', 'b300'
    
    # Training mode selection
    use_advanced_infrastructure = TRAINING_INFRASTRUCTURE_AVAILABLE  # Set to False to force basic trainer
    
    # Training parameters
    training_params = {
        'use_moe': True,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'min_lr': 1e-6,
        'lr_scheduler': "cosine",
        'batch_size': 20,
        'gradient_accumulation_steps': 8,
        'precision': "fp16",
        'inference_precision': "int8",
        'compile': True,
        'max_memory_usage': 0.85,
        'save_every_n_batches': 1000,
        'eval_every_n_batches': 500,
        'use_flash_attention': True,
        'gradient_checkpointing': True,
        'num_workers': 2,
        'save_total_limit': 5,
        'weight_decay': 0.01,
    }
    
    # Data configuration
    data_params = {
        'train_data_path': 'oasst1_data/oasst1_train.jsonl',
        'eval_data_path': 'oasst1_data/oasst1_train.jsonl',
        'max_conversations_per_file': 50000,
        'create_dummy_data': False,
        'dummy_data_samples': 200,
    }
    
    # DeepSpeed configuration
    deepspeed_params = {
        'use_deepspeed': False,
        'cpu_offload': False,
        'cpu_offload_optimizer': False,
        'cpu_offload_parameters': False,
        'zero_stage': 2,
        'nvme_path': None,
        'max_grad_norm': 1.0,
    }
    
    # Quantization configuration (for advanced infrastructure)
    quantization_params = {
        'quantization_method': 'bnb',  # Options: None, 'bnb', 'gptq', 'quanto'
        'quantization_bits': 8,  # Options: None, 4, 8
    }
    
    # Monitoring and logging
    monitoring_params = {
        'log_level': "INFO",
        'experiment_name': f'Production_Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'early_stopping_patience': 5,
        'backup_every_n_hours': 12,
        'enable_wandb': False,
        'wandb_project': None,
        'wandb_entity': None,
        'health_check_interval': 50,
        'log_every_n_steps': 50,
    }
    
    # =================================================
    # END CONFIGURATION SECTION
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, monitoring_params['log_level']),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    print("\n" + "="*80)
    if use_advanced_infrastructure and TRAINING_INFRASTRUCTURE_AVAILABLE:
        print("ADAPTIVE AI-DRIVEN TRAINING WITH SELF-IMPROVEMENT")
    else:
        print("PRODUCTION DEEPSEEK MOE TRANSFORMER TRAINING")
    print("="*80)
    print(f"Experiment: {monitoring_params['experiment_name']}")
    print(f"Advanced Infrastructure: {use_advanced_infrastructure and TRAINING_INFRASTRUCTURE_AVAILABLE}")
    print(f"DeepSpeed Available: {DEEPSPEED_AVAILABLE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            print(f"  GPU {i}: {gpu_name} (Compute: {capability[0]}.{capability[1]})")
    print("="*80)
    
    try:
        # Step 1: Create base configuration
        print(f"\nStep 1: Creating base configuration ({config_choice})")
        if hasattr(ConfigPresets, config_choice):
            config = getattr(ConfigPresets, config_choice)()
        else:
            raise ValueError(f"Unknown config preset: {config_choice}")
        
        # Step 2: Apply all parameter overrides
        all_params = {
            **training_params, 
            **data_params, 
            **deepspeed_params, 
            **quantization_params,
            **monitoring_params
        }
        
        print(f"Step 2: Applying {len(all_params)} parameter overrides")
        for key, value in all_params.items():
            if value is not None:
                old_value = getattr(config, key, None)
                setattr(config, key, value)
                if old_value is not None and old_value != value:
                    print(f"  {key}: {old_value} -> {value}")
        
        # Step 2.5: Validate precision support
        print(f"\nStep 2.5: Validating precision support")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        training_precision = training_params.get('precision', 'fp32')
        inference_precision = training_params.get('inference_precision', 'fp16')
        
        # Validate training precision
        is_supported, error_msg = validate_precision_support(training_precision, device)
        if not is_supported:
            print(f"\n{'='*80}")
            print("ERROR: UNSUPPORTED PRECISION FOR TRAINING")
            print(f"{'='*80}")
            print(f"{error_msg}")
            print(f"{'='*80}\n")
            return 1
        else:
            print(f"  Training precision '{training_precision}' is supported on {device.type}")
        
        # Validate inference precision
        is_supported, error_msg = validate_precision_support(inference_precision, device)
        if not is_supported:
            print(f"  WARNING: Inference precision '{inference_precision}' is not supported.")
            print(f"  {error_msg}")
            print(f"  Inference will use training precision instead.")
            config.inference_precision = training_precision
        else:
            print(f"  Inference precision '{inference_precision}' is supported on {device.type}")
        
        # Step 3: Validate configuration
        print(f"\nStep 3: Validating configuration")
        config.validate()
        print("Configuration validation passed")
        
        # Step 4: Initialize tokenizer
        print(f"\nStep 4: Initializing tokenizer")
        tokenizer = ConversationTokenizer()
        config.vocab_size = tokenizer.vocab_size
        print(f"Tokenizer initialized with vocab_size: {config.vocab_size}")
        
        # Step 5: Initialize model
        print(f"\nStep 5: Initializing model")
        model_config = config_to_deepseek_config(config)
        model = DeepSeekTransformer(model_config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Step 6: Setup datasets
        print(f"\nStep 6: Setting up datasets")
        
        # Handle dummy data creation
        train_data_path = Path(data_params['train_data_path'])
        if data_params.get('create_dummy_data', False) or not train_data_path.exists():
            if not train_data_path.exists():
                print(f"Training data not found at {train_data_path}")
            print(f"Creating dummy training data...")
            create_dummy_training_data(train_data_path, data_params.get('dummy_data_samples', 200))
        
        # Create datasets
        train_dataset = ConversationDataset(
            str(train_data_path), tokenizer, config, "train"
        )
        print(f"Training dataset loaded: {len(train_dataset)} samples")
        
        eval_dataset = None
        eval_data_path = Path(data_params['eval_data_path'])
        if eval_data_path.exists() and eval_data_path != train_data_path:
            eval_dataset = ConversationDataset(
                str(eval_data_path), tokenizer, config, "eval"
            )
            print(f"Evaluation dataset loaded: {len(eval_dataset)} samples")
        else:
            print("Using training data for evaluation (same file)")
            eval_dataset = train_dataset
        
        # Step 7: Initialize training system
        print(f"\nStep 7: Initializing training system")
        
        trainer = None
        orchestrator = None
        
        if use_advanced_infrastructure and TRAINING_INFRASTRUCTURE_AVAILABLE:
            # Use advanced adaptive training orchestrator
            print("Using AdaptiveTrainingOrchestrator with AI-driven optimization")
            orchestrator = AdaptiveTrainingOrchestrator(config)
            
            # The orchestrator will initialize its own components
            print("Adaptive training system will be initialized by orchestrator")
            print(f"  Meta-learning enabled: Yes")
            print(f"  Real-time monitoring: Yes")
            print(f"  Adaptive hyperparameter optimization: Yes")
            print(f"  Architecture evolution: {'Yes' if config.use_moe else 'No'}")
            
        else:
            # Use basic ProductionTrainer
            print("Using ProductionTrainer (basic training)")
            logger = logging.getLogger(__name__)
            trainer = ProductionTrainer(model, tokenizer, config, logger)
            print(f"Trainer initialized: {'DeepSpeed' if trainer.use_deepspeed else 'Standard PyTorch'}")
        
        # Step 8: Start training
        print(f"\nStep 8: Starting training")
        print(f"Configuration:")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Precision: {config.precision}")
        print(f"  Inference precision: {getattr(config, 'inference_precision', 'same as training')}")
        print(f"  MoE enabled: {config.use_moe}")
        if config.use_moe:
            print(f"  Experts: {config.num_experts}, Top-K: {config.moe_top_k}")
        if quantization_params.get('quantization_method'):
            print(f"  Quantization: {quantization_params['quantization_method']} "
                  f"{quantization_params['quantization_bits']}-bit")
        
        # Create experiment directory
        experiment_dir = Path(f"experiments/{config.experiment_name}")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial configuration
        config_path = experiment_dir / "config.yaml"
        config.save(str(config_path))
        print(f"Configuration saved: {config_path}")
        
        # Run training
        if orchestrator:
            # Use adaptive training orchestrator
            orchestrator.run_adaptive_training()
        elif trainer:
            # Use standard trainer
            trainer.train(train_dataset, eval_dataset)
        else:
            raise RuntimeError("No training system initialized")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Experiment: {config.experiment_name}")
        
        if trainer:
            print(f"Total steps: {trainer.global_step}")
            print(f"Final epoch: {trainer.current_epoch}")
            print(f"Best eval loss: {trainer.best_eval_loss:.6f}")
        
        if orchestrator:
            status = orchestrator.get_adaptive_status()
            print(f"Adaptive decisions made: {status.get('adaptive_decisions_made', 'N/A')}")
            print(f"Metrics collected: {status.get('metrics_collected', 'N/A')}")
            print(f"Meta-learning runs: {status.get('meta_learning_runs', 'N/A')}")
        
        # Save final summary
        summary = {
            'experiment_name': config.experiment_name,
            'training_completed': datetime.now().isoformat(),
            'model_parameters': {
                'total': total_params,
                'trainable': trainable_params,
            },
            'configuration': {
                'model': config_choice,
                'moe_enabled': config.use_moe,
                'quantization': quantization_params.get('quantization_method'),
                'precision': config.precision,
                'training_mode': 'adaptive' if orchestrator else 'standard'
            }
        }
        
        if trainer:
            summary['training_state'] = {
                'total_steps': trainer.global_step,
                'epochs_completed': trainer.current_epoch,
                'best_eval_loss': trainer.best_eval_loss,
            }
        
        if orchestrator:
            try:
                summary['adaptive_training'] = orchestrator.get_adaptive_status()
            except Exception as e:
                print(f"Could not get orchestrator status: {e}")
        
        summary_path = experiment_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Training summary saved: {summary_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
        # Save emergency state
        if orchestrator:
            try:
                orchestrator._save_meta_learning_state()
            except:
                pass
        
        return 0
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        print("\nCleaning up resources...")
        
        if 'orchestrator' in locals() and orchestrator:
            try:
                orchestrator.cleanup()
            except Exception as e:
                print(f"Orchestrator cleanup error: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("Cleanup complete")
    
    return 0


if __name__ == "__main__":
    exit(main())