"""
Production Logging and Monitoring Module
Enhanced logging with structured format and monitoring backends.
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Optional monitoring imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class ProductionLogger:
    """Enhanced logging with structured format and monitoring."""
    
    def __init__(self, log_level: str = "INFO", experiment_name: str = None):
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(f"logs/{self.experiment_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup structured logging
        log_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_dir / "training.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        file_handler.setFormatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        
        # Configure root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
        
        # Performance metrics
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.system_stats = []
        self.training_stats = []
        
        # Initialize monitoring
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup system monitoring."""
        # Initialize wandb if available
        if HAS_WANDB:
            try:
                wandb.init(
                    project="conversational-transformer",
                    name=self.experiment_name,
                    dir=str(self.log_dir)
                )
                self.use_wandb = True
            except Exception as e:
                logging.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False
        
        # Initialize tensorboard if available
        if HAS_TENSORBOARD:
            try:
                self.tb_writer = SummaryWriter(str(self.log_dir / "tensorboard"))
                self.use_tensorboard = True
            except Exception as e:
                logging.warning(f"Failed to initialize tensorboard: {e}")
                self.use_tensorboard = False
        else:
            self.use_tensorboard = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """Log metrics to all available backends."""
        timestamp = datetime.now().isoformat()
        
        # Prepare metrics with prefix
        prefixed_metrics = {}
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            prefixed_metrics[full_key] = value
        
        # Log to file
        metric_entry = {
            "timestamp": timestamp,
            "step": step,
            "metrics": prefixed_metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric_entry) + '\n')
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(prefixed_metrics, step=step)
        
        # Log to tensorboard
        if self.use_tensorboard:
            for key, value in prefixed_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
    
    def log_system_stats(self, step: int):
        """Log system performance statistics."""
        try:
            import psutil
            import torch
            
            stats = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / 1e9,
            }
            
            if torch.cuda.is_available():
                stats.update({
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                })
            
            self.log_metrics(stats, step, "system")
        except Exception as e:
            logging.warning(f"Failed to log system stats: {e}")
    
    def close(self):
        """Close all logging backends."""
        if self.use_wandb:
            wandb.finish()
        
        if self.use_tensorboard:
            self.tb_writer.close()


class TrainingHealthMonitor:
    """Monitor training health and detect anomalies."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_history = []
        self.grad_norm_history = []
        self.consecutive_nan_count = 0
        self.consecutive_high_loss_count = 0
        
    def update(self, loss: float, grad_norm: float):
        """Update monitoring with new metrics."""
        import math
        import numpy as np
        
        # Check for NaN/Inf
        if math.isnan(loss) or math.isinf(loss):
            self.consecutive_nan_count += 1
        else:
            self.consecutive_nan_count = 0
            
        # Update history
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        
        # Maintain window size
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            self.grad_norm_history.pop(0)
        
        # Check for divergence
        if len(self.loss_history) > 10:
            recent_avg = np.mean(self.loss_history[-10:])
            if len(self.loss_history) > 20:
                earlier_avg = np.mean(self.loss_history[-20:-10])
                if recent_avg > earlier_avg * 1.5:
                    self.consecutive_high_loss_count += 1
                else:
                    self.consecutive_high_loss_count = 0
    
    def get_status(self) -> str:
        """Get current health status."""
        if self.consecutive_nan_count > 3:
            return "CRITICAL"
        elif self.consecutive_high_loss_count > 10:
            return "WARNING"
        elif len(self.grad_norm_history) > 0 and self.grad_norm_history[-1] > 100:
            return "WARNING"
        else:
            return "HEALTHY"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        import numpy as np
        
        return {
            'total_nan_episodes': self.consecutive_nan_count,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0,
            'avg_grad_norm': np.mean(self.grad_norm_history) if self.grad_norm_history else 0,
            'loss_std': np.std(self.loss_history) if len(self.loss_history) > 1 else 0,
            'status': self.get_status()
        }