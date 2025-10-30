# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import json
import logging
import math
import signal
import time
import traceback
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from collections import deque
import threading
import queue

import numpy as np
import torch
import torch.nn.functional as F

@dataclass
class TrainingMetrics:
    """Comprehensive training metrics for adaptive intelligence."""
    epoch: int
    step: int
    loss: float
    grad_norm: float
    learning_rate: float
    expert_utilization: Dict[str, float]
    memory_usage: Dict[str, float]
    throughput: float
    semantic_coherence: float
    factual_accuracy: float
    reasoning_score: float
    timestamp: datetime
    
    def to_dict(self):
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class AdaptiveDecision:
    """Represents an adaptive decision made by the intelligence system."""
    decision_type: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    expected_improvement: float
    timestamp: datetime

class MetaLearningEngine:
    """Learns how to train more effectively over time."""
    
    def __init__(self, orchestrator=None):
        self.training_history = []
        self.successful_strategies = []
        self.meta_model = None
        self.adaptation_buffer = deque(maxlen=1000)
        self.orchestrator = orchestrator  # Store reference to get model params

    def _synthesize_suggestions(self, successful_patterns, current_metrics):
        """Synthesize hyperparameter suggestions from successful patterns."""
        if not successful_patterns:
            return {}

        # Average successful hyperparameters
        avg_lr = np.mean([p['config'].get('learning_rate', self.orchestrator.config.learning_rate if self.orchestrator else 0.001) 
                            for p in successful_patterns])
        
        suggestions = {
            'learning_rate': {
                'value': avg_lr,
                'confidence': min(len(successful_patterns) / 10.0, 0.9)
            }
        }

        # Add batch size suggestions if available
        batch_sizes = [p['config'].get('batch_size') for p in successful_patterns if 'batch_size' in p['config']]
        if batch_sizes:
            avg_batch_size = int(np.mean(batch_sizes))
            suggestions['batch_size'] = {
                'value': avg_batch_size,
                'confidence': min(len(batch_sizes) / 10.0, 0.8)
            }

        return suggestions
        
    def record_training_outcome(self, config, metrics, final_performance):
        """Record the outcome of a training run for meta-learning."""
        outcome = {
            'config': self._serialize_config(config),
            'metrics_progression': [m.to_dict() for m in metrics],
            'final_performance': final_performance,
            'training_duration': len(metrics),
            'success_score': self._calculate_success_score(metrics, final_performance)
        }
        self.training_history.append(outcome)
        self._update_meta_model()
    
    def _update_meta_model(self):
        """Update the meta-learning model based on training history."""
        # This is a placeholder for future meta-learning implementation
        # For now, just track successful strategies
        if len(self.training_history) > 0:
            recent_run = self.training_history[-1]
            if recent_run['success_score'] > 0.7:
                # Extract successful hyperparameters
                strategy = {
                    'learning_rate': recent_run['config'].get('learning_rate'),
                    'batch_size': recent_run['config'].get('batch_size'),
                    'success_score': recent_run['success_score'],
                    'timestamp': time.time()
                }
                self.successful_strategies.append(strategy)
                
                # Keep only top 20 strategies
                self.successful_strategies.sort(key=lambda x: x['success_score'], reverse=True)
                self.successful_strategies = self.successful_strategies[:20]
    
    def suggest_hyperparameters(self, current_metrics, config):
        """Suggest hyperparameter adjustments based on meta-learning."""
        if len(self.training_history) < 3:
            return self._conservative_suggestions(current_metrics)

        # Get model params from orchestrator
        current_params = 0
        current_device = 'cpu'
        if self.orchestrator and self.orchestrator.model:
            current_params = sum(p.numel() for p in self.orchestrator.model.parameters())
            current_device = str(self.orchestrator.device.type)

        # Find similar training scenarios
        similar_runs = self._find_similar_runs(current_metrics, config, current_params, current_device)

        # Extract successful patterns
        successful_patterns = [run for run in similar_runs if run['success_score'] > 0.7]

        if not successful_patterns:
            return self._exploratory_suggestions(current_metrics)

        # Generate suggestions based on successful patterns
        suggestions = self._synthesize_suggestions(successful_patterns, current_metrics)

        return suggestions
    
    def _conservative_suggestions(self, current_metrics):
        """Conservative hyperparameter suggestions for cold start."""
        return {
            'learning_rate': {'value': current_metrics.learning_rate * 0.9, 'confidence': 0.5},
            'batch_size': {'value': None, 'confidence': 0.3}  # Don't change batch size conservatively
        }
    
    def _exploratory_suggestions(self, current_metrics):
        """Exploratory suggestions when no similar runs found."""
        return {
            'learning_rate': {'value': current_metrics.learning_rate * 1.1, 'confidence': 0.4},
            'warmup_steps': {'value': 500, 'confidence': 0.5}
        }
    
    def _find_similar_runs(self, current_metrics, config, current_model_params, current_device):
        """Find training runs with similar characteristics using multi-dimensional similarity."""
        similar = []

        for run in self.training_history:
            if len(run['metrics_progression']) == 0:
                continue
                
            similarity_score = self._calculate_run_similarity(
                current_metrics, 
                run, 
                current_model_params,
                current_device,
                config  # FIX: Pass config explicitly
            )

            # Use threshold of 0.6 for similarity
            if similarity_score > 0.6:
                similar.append((run, similarity_score))

        # Return sorted by similarity (most similar first)
        similar.sort(key=lambda x: x[1], reverse=True)
        return [run for run, score in similar]
    
    def _calculate_run_similarity(self, current_metrics, historical_run, current_params, current_device, config):
        """Calculate multi-dimensional similarity score between current and historical runs."""
        score = 0.0
    
        # Loss similarity (weight: 0.4)
        initial_loss = historical_run['metrics_progression'][0].get('loss', float('inf'))
        if initial_loss < float('inf'):
            loss_diff = abs(current_metrics.loss - initial_loss)
            loss_similarity = max(0, 1.0 - loss_diff / 5.0)  # Normalize by max expected diff
            score += 0.4 * loss_similarity

        # Model size similarity (weight: 0.3)
        if 'model_params' in historical_run and current_params > 0:
            hist_params = historical_run['model_params']
            size_ratio = min(current_params, hist_params) / max(current_params, hist_params)
            score += 0.3 * size_ratio

        # Hardware similarity (weight: 0.2)
        if historical_run.get('device_type') == current_device:
            score += 0.2

        # Architecture similarity (weight: 0.1) - FIX: Use passed config parameter
        if historical_run['config'].get('use_moe') == getattr(config, 'use_moe', False):
            score += 0.05
        if historical_run['config'].get('use_mod') == getattr(config, 'use_mod', False):
            score += 0.05

        return score
    
    def predict_training_trajectory(self, current_metrics, config):
        """Predict how training will progress."""
        if len(self.adaptation_buffer) < 10:
            return None
        
        recent_metrics = list(self.adaptation_buffer)[-10:]
        loss_trend = np.polyfit(range(len(recent_metrics)), 
                               [m.loss for m in recent_metrics], 1)[0]
        
        # Predict plateau, convergence, or divergence
        if abs(loss_trend) < 1e-4:
            return {
                'prediction': 'plateau',
                'confidence': 0.8,
                'suggested_action': 'increase_lr_or_change_architecture',
                'expected_improvement': 0.1
            }
        elif loss_trend < -1e-3:
            return {
                'prediction': 'healthy_convergence',
                'confidence': 0.9,
                'suggested_action': 'continue',
                'expected_improvement': abs(loss_trend) * 100
            }
        else:
            return {
                'prediction': 'potential_divergence',
                'confidence': 0.7,
                'suggested_action': 'reduce_lr_or_add_regularization',
                'expected_improvement': 0.05
            }
    
    def _serialize_config(self, config):
        """Convert config to serializable format."""
        return {
            attr: getattr(config, attr) for attr in dir(config) 
            if not attr.startswith('_') and not callable(getattr(config, attr))
        }
    
    def _calculate_success_score(self, metrics, final_performance):
        """Calculate how successful a training run was."""
        if not metrics:
            return 0.0
        
        # Factors: convergence speed, final performance, stability
        convergence_speed = 1.0 / len(metrics) if len(metrics) > 0 else 0
        stability = 1.0 - np.std([m.loss for m in metrics[-10:]])
        
        return 0.4 * final_performance + 0.3 * convergence_speed + 0.3 * stability

class AdaptiveHyperparameterOptimizer:
    """Continuously optimizes hyperparameters during training."""
    
    def __init__(self):
        self.optimization_history = []
        self.current_search_space = {}
        self.performance_buffer = deque(maxlen=50)
        self.last_adjustment_step = 0
        
    def should_adjust_learning_rate(self, current_metrics):
        """Decide whether to adjust learning rate."""

        if len(self.performance_buffer) > 0:
            steps_since_last = current_metrics.step - self.last_adjustment_step
            if steps_since_last < 50:  # Don't adjust too often
                self.performance_buffer.append(current_metrics)
                return None

        self.performance_buffer.append(current_metrics)
        recent_losses = [m.loss for m in list(self.performance_buffer)[-20:]]
        very_recent = [m.loss for m in list(self.performance_buffer)[-5:]]

        # 1. PLATEAU - If loss barely changing
        if np.std(very_recent) < 0.01 and np.mean(very_recent) > 0.5:
            self.last_adjustment_step = current_metrics.step
            return {
                'action': 'increase',
                'factor': 1.5,
                'reasoning': f'Loss plateau: std={np.std(very_recent):.4f}',
                'emergency': False,
            }

        # 2. DIVERGENCE - If loss increasing
        recent_mean = np.mean(very_recent)
        older_mean = np.mean(recent_losses[-15:-10]) if len(recent_losses) >= 15 else recent_mean
        if recent_mean > older_mean + 0.3:
            self.last_adjustment_step = current_metrics.step
            return {
                'action': 'decrease',
                'factor': 0.5,
                'reasoning': f'Loss increasing: {older_mean:.3f} → {recent_mean:.3f}',
                'emergency': False
            }

        # 3. GOOD PROGRESS - If steadily decreasing
        if recent_mean < older_mean - 0.1 and np.std(very_recent) < 0.05:
            self.last_adjustment_step = current_metrics.step
            return {
                'action': 'increase',
                'factor': 1.2,
                'reasoning': 'Steady improvement, accelerating'
            }
        # Check for instability
        grad_norms = [m.grad_norm for m in list(self.performance_buffer)[-5:]]
        if np.mean(grad_norms) > 10.0:
            return {
                'action': 'decrease',
                'factor': 0.7,
                'reasoning': 'High gradient norms detected, reducing LR for stability',
                'emergency': False
            }
        
        return None
    
    def optimize_batch_size(self, current_metrics, memory_usage):
        """Dynamically optimize batch size based on performance and memory."""
        current_memory_usage = memory_usage.get('gpu_memory_percent', 0)
        
        # If memory usage is low and performance is good, increase batch size
        if current_memory_usage < 70 and current_metrics.loss < 2.0:
            return {
                'action': 'increase',
                'new_size': int(current_metrics.step * 1.25),
                'reasoning': 'Low memory usage and good performance, increasing batch size'
            }
        
        # If memory usage is high, decrease batch size
        if current_memory_usage > 90:
            return {
                'action': 'decrease',
                'new_size': max(1, int(current_metrics.step * 0.8)),
                'reasoning': 'High memory usage, reducing batch size'
            }
        
        return None

class ArchitectureEvolution:
    """Handles dynamic architecture changes during training."""
    
    def __init__(self):
        self.architecture_history = []
        self.performance_tracking = {}
        
    def should_add_expert(self, expert_utilization, performance_metrics):
        """Decide whether to add a new MoE expert."""
        if not expert_utilization:
            return None
            
        # Check if current experts are overutilized
        max_utilization = max(expert_utilization.values())
        avg_utilization = np.mean(list(expert_utilization.values()))
        
        if max_utilization > 0.9 and avg_utilization > 0.7:
            return {
                'action': 'add_expert',
                'expert_type': 'general',
                'reasoning': f'High expert utilization (max: {max_utilization:.2f}, avg: {avg_utilization:.2f})',
                'expected_improvement': 0.1
            }
        
        return None
    
    def should_prune_expert(self, expert_utilization, performance_metrics):
        """Decide whether to remove underutilized experts."""
        if not expert_utilization or len(expert_utilization) <= 2:
            return None
            
        min_utilization = min(expert_utilization.values())
        underutilized_experts = [k for k, v in expert_utilization.items() if v < 0.1]
        
        if len(underutilized_experts) > 0 and min_utilization < 0.05:
            return {
                'action': 'prune_expert',
                'expert_id': min(expert_utilization, key=expert_utilization.get),
                'reasoning': f'Expert underutilized: {min_utilization:.3f}',
                'expected_improvement': 0.02
            }
        
        return None
    
    def suggest_architecture_changes(self, current_metrics, model_info):
        """Suggest architecture modifications based on current performance."""
        suggestions = []
        
        # Check expert utilization
        if hasattr(current_metrics, 'expert_utilization'):
            expert_suggestion = self.should_add_expert(
                current_metrics.expert_utilization, current_metrics
            )
            if expert_suggestion:
                suggestions.append(expert_suggestion)
            
            prune_suggestion = self.should_prune_expert(
                current_metrics.expert_utilization, current_metrics
            )
            if prune_suggestion:
                suggestions.append(prune_suggestion)
        
        return suggestions

class RealTimeAnalytics:
    def __init__(self):
        self.metrics_buffer = deque(maxlen=1000)
        self.anomaly_detector = None
        self.trend_analyzer = None
        
        # Configurable thresholds
        self.anomaly_thresholds = {
            'loss_spike_std_multiplier': 2.0,
            'loss_spike_min_increase': 0.1,
            'gradient_explosion_threshold': 100.0,
            'gradient_explosion_relative': 10.0,
            'min_buffer_size': 50,
            'recent_window': 10,
        }
    
    # FIX: Move this method INSIDE the class
    def update_anomaly_thresholds(self, threshold_name: str, new_value: float):
        """Allow dynamic threshold adjustment."""
        if threshold_name in self.anomaly_thresholds:
            old_value = self.anomaly_thresholds[threshold_name]
            self.anomaly_thresholds[threshold_name] = new_value
            logging.info(f"Updated anomaly threshold '{threshold_name}': {old_value} -> {new_value}")
        else:
            logging.warning(f"Unknown threshold name: {threshold_name}")

    def _predict_convergence(self, coeffs, current_step):
        """Predict when training will converge."""
        # Simple quadratic extrapolation
        future_steps = np.arange(current_step, current_step + 1000, 10)
        future_losses = np.polyval(coeffs, future_steps)

        # Find when loss stops decreasing significantly
        derivatives = np.diff(future_losses)
        convergence_point = np.where(np.abs(derivatives) < 1e-4)[0]

        if len(convergence_point) > 0:
            return int(future_steps[convergence_point[0]])

        return None
        
    def analyze_loss_dynamics(self, recent_metrics):
        """Analyze loss curve dynamics for insights."""
        if len(recent_metrics) < 10:
            return None
        
        losses = [m.loss for m in recent_metrics]
        steps = [m.step for m in recent_metrics]
        
        # Fit polynomial to detect trends
        coeffs = np.polyfit(steps, losses, 2)
        
        # Analyze curvature
        curvature = coeffs[0]
        trend = coeffs[1]
        
        insights = {
            'trend_direction': 'decreasing' if trend < 0 else 'increasing',
            'trend_strength': abs(trend),
            'curvature': 'concave_up' if curvature > 0 else 'concave_down',
            'predicted_convergence': self._predict_convergence(coeffs, steps[-1])
        }
        
        return insights
    
    def detect_training_anomalies(self, current_metrics):
        """Detect unusual patterns in training using adaptive thresholds."""
        if len(self.metrics_buffer) < self.anomaly_thresholds['min_buffer_size']:
            self.metrics_buffer.append(current_metrics)
            return None

        self.metrics_buffer.append(current_metrics)

        # Configurable windows
        recent_window = self.anomaly_thresholds['recent_window']
        recent_losses = [m.loss for m in list(self.metrics_buffer)[-recent_window:]]
        historical_losses = [m.loss for m in list(self.metrics_buffer)[-50:-recent_window]]

        if not historical_losses:
            return None

        recent_mean = np.mean(recent_losses)
        historical_mean = np.mean(historical_losses)
        historical_std = np.std(historical_losses)

        anomalies = []

        # Adaptive loss spike detection
        std_multiplier = self.anomaly_thresholds['loss_spike_std_multiplier']
        min_increase = self.anomaly_thresholds['loss_spike_min_increase']

        threshold = historical_mean + std_multiplier * historical_std
        absolute_increase = recent_mean - historical_mean

        if recent_mean > threshold and absolute_increase > min_increase:
            severity = 'critical' if absolute_increase > 1.0 else 'high'
            anomalies.append({
                'type': 'loss_spike',
                'severity': severity,
                'description': f'Loss increased significantly: {recent_mean:.3f} vs {historical_mean:.3f} (+{absolute_increase:.3f})',
                'relative_increase': absolute_increase / historical_mean
            })

        # Adaptive gradient explosion detection
        abs_threshold = self.anomaly_thresholds['gradient_explosion_threshold']
        relative_threshold = self.anomaly_thresholds['gradient_explosion_relative']

        # Calculate historical gradient norm mean
        historical_grad_norms = [m.grad_norm for m in list(self.metrics_buffer)[-50:-recent_window] if m.grad_norm > 0]

        is_explosion = current_metrics.grad_norm > abs_threshold
        
        if historical_grad_norms:
            hist_grad_mean = np.mean(historical_grad_norms)
            is_explosion = is_explosion or (current_metrics.grad_norm > hist_grad_mean * relative_threshold)
    
        if is_explosion:
            anomalies.append({
                'type': 'gradient_explosion',
                'severity': 'critical',
                'description': f'Gradient norm extremely high: {current_metrics.grad_norm:.2f}',
                'threshold_used': abs_threshold
            })

        # New: Detect expert collapse in MoE
        if hasattr(current_metrics, 'expert_utilization') and current_metrics.expert_utilization:
            expert_usage = list(current_metrics.expert_utilization.values())
            if expert_usage:
                max_usage = max(expert_usage)
                min_usage = min(expert_usage)

                if min_usage < 0.01 and max_usage > 0.5:
                    anomalies.append({
                        'type': 'expert_collapse',
                        'severity': 'high',
                        'description': f'Expert imbalance detected: min={min_usage:.1%}, max={max_usage:.1%}'
                    })

        return anomalies if anomalies else None

class ProductionMonitoring:
    """Advanced monitoring for production deployment."""
    
    def __init__(self):
        self.performance_tracker = {}
        self.safety_monitor = {}
        self.efficiency_tracker = {}
        
    def monitor_semantic_drift(self, generated_texts, reference_corpus):
        """Monitor for semantic drift in generated content."""
        # Placeholder for semantic similarity analysis
        drift_score = np.random.random()
        
        if drift_score < 0.7:
            return {
                'alert': 'semantic_drift',
                'severity': 'medium',
                'score': drift_score,
                'recommendation': 'Consider fine-tuning with recent data'
            }
        return None
    
    def track_safety_metrics(self, generated_content):
        """Track safety and bias metrics."""
        # Placeholder for toxicity/bias detection
        safety_scores = {
            'toxicity': np.random.random(),
            'bias_gender': np.random.random(),
            'bias_racial': np.random.random(),
            'factual_accuracy': np.random.random()
        }
        
        alerts = []
        for metric, score in safety_scores.items():
            if score < 0.8:
                alerts.append({
                    'metric': metric,
                    'score': score,
                    'severity': 'high' if score < 0.6 else 'medium'
                })
        
        return alerts if alerts else None

class AdaptiveTrainingOrchestrator:
    """Enhanced orchestrator with adaptive intelligence and self-improvement."""
    
    def __init__(self, config):
        self.config = config
        self.base_config = self._deep_copy_config(config)
        
        # Initialize logger
        try:
            from monitoring.logger import ProductionLogger
            self.logger = ProductionLogger(config.log_level, config.experiment_name)
        except ImportError:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        
        # Set seeds
        self._set_seeds(getattr(config, 'seed', 42))
        
        # Create experiment directory
        self.experiment_dir = Path(f"experiments/{config.experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Adaptive intelligence components
        self.meta_learner = MetaLearningEngine(orchestrator=self)
        self.hyperparameter_optimizer = AdaptiveHyperparameterOptimizer()
        self.architecture_evolution = ArchitectureEvolution()
        self.analytics = RealTimeAnalytics()
        self.production_monitor = ProductionMonitoring()
        
        # Training state
        self.training_metrics_history = []
        self.adaptive_decisions = []
        self.current_metrics = None
        self.is_training = False
        self.should_stop = False
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Real-time monitoring thread
        self.monitoring_thread = None
        self.monitoring_queue = queue.Queue()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Load previous meta-learning data if available
        self._load_meta_learning_state()
        
        # Save initial configuration
        if hasattr(config, 'save'):
            config.save(str(self.experiment_dir / "initial_config.yaml"))
        
        logging.info("Adaptive Training Orchestrator initialized with AI-driven optimization")
    
    def _deep_copy_config(self, config):
        """Create a deep copy of config for comparison."""
        import copy
        return copy.deepcopy(config)
    
    def _setup_trainer_scheduler(self, train_dataset):
        """Setup learning rate scheduler for the trainer - FIXED."""
        if not self.trainer:
            logging.error("Cannot setup scheduler: trainer not initialized")
            return

        # Calculate total training steps
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        batches_per_epoch = len(train_dataset) // self.config.batch_size
        steps_per_epoch = batches_per_epoch // gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs

        logging.info(f"Setting up scheduler:")
        logging.info(f"  Batches per epoch: {batches_per_epoch}")
        logging.info(f"  Steps per epoch: {steps_per_epoch}")
        logging.info(f"  Total steps: {total_steps}")

        # ✅ FIX: Pass total_steps to trainer's setup method
        if hasattr(self.trainer, '_setup_scheduler'):
            self.trainer._setup_scheduler(total_steps)
            if self.trainer.scheduler:
                logging.info(f"✅ Scheduler initialized: {type(self.trainer.scheduler).__name__}")

                # ✅ CRITICAL: Store scheduler reference in orchestrator too
                self.scheduler = self.trainer.scheduler
                logging.info("✅ Scheduler reference stored in orchestrator")
        else:
            logging.warning("Trainer does not have _setup_scheduler method")

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, saving adaptive learning state...")
            self.should_stop = True
            self._save_meta_learning_state()
            if self.trainer and hasattr(self.trainer, 'should_stop'):
                self.trainer.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _load_meta_learning_state(self):
        """Load previous meta-learning data."""
        meta_state_path = self.experiment_dir / "meta_learning_state.pkl"
        if meta_state_path.exists():
            try:
                with open(meta_state_path, 'rb') as f:
                    state = pickle.load(f)
                    self.meta_learner.training_history = state.get('training_history', [])
                    self.meta_learner.successful_strategies = state.get('successful_strategies', [])
                    logging.info(f"Loaded {len(self.meta_learner.training_history)} previous training runs for meta-learning")
            except Exception as e:
                logging.warning(f"Could not load meta-learning state: {e}")
    
    def _save_meta_learning_state(self):
        """Save meta-learning state for future runs."""
        try:
            state = {
                'training_history': self.meta_learner.training_history,
                'successful_strategies': self.meta_learner.successful_strategies,
                'adaptive_decisions': self.adaptive_decisions,
                'timestamp': datetime.now()
            }
            
            meta_state_path = self.experiment_dir / "meta_learning_state.pkl"
            with open(meta_state_path, 'wb') as f:
                pickle.dump(state, f)
                
            # Also save human-readable summary
            summary_path = self.experiment_dir / "adaptive_learning_summary.json"
            summary = {
                'total_training_runs': len(self.meta_learner.training_history),
                'successful_adaptations': len([d for d in self.adaptive_decisions if d.confidence > 0.7]),
                'top_strategies': self.meta_learner.successful_strategies[:10],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logging.info(f"Saved meta-learning state with {len(self.meta_learner.training_history)} training runs")
        except Exception as e:
            logging.error(f"Failed to save meta-learning state: {e}")
    
    def start_real_time_monitoring(self):
        """Start real-time monitoring thread."""
        def monitoring_loop():
            while self.is_training and not self.should_stop:
                try:
                    # Get latest metrics
                    if not self.monitoring_queue.empty():
                        metrics = self.monitoring_queue.get_nowait()
                        self._process_real_time_metrics(metrics)
                    
                    time.sleep(1)  # Check every second
                except Exception as e:
                    logging.error(f"Error in monitoring loop: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logging.info("Started real-time monitoring thread")
    
    def _process_real_time_metrics(self, metrics):
        """Process metrics in real-time and make adaptive decisions."""
        self.current_metrics = metrics
        self.training_metrics_history.append(metrics)
        self.analytics.metrics_buffer.append(metrics)
        
        # Check for anomalies
        anomalies = self.analytics.detect_training_anomalies(metrics)
        if anomalies:
            for anomaly in anomalies:
                logging.warning(f"Training anomaly detected: {anomaly}")
                self._handle_training_anomaly(anomaly)

        if self.global_step % 100 == 0:
            scheduler_status = self.get_scheduler_status()
            logging.info(f"📊 Scheduler Status at step {self.global_step}:")
            for key, value in scheduler_status.items():
                logging.info(f"   {key}: {value}")
        
        # Analyze loss dynamics
        if len(self.training_metrics_history) >= 10:
            insights = self.analytics.analyze_loss_dynamics(self.training_metrics_history[-10:])
            if insights:
                self._act_on_loss_insights(insights)
        
        # Check if hyperparameters should be adjusted
        lr_adjustment = self.hyperparameter_optimizer.should_adjust_learning_rate(metrics)
        if lr_adjustment:
            self._apply_learning_rate_adjustment(lr_adjustment)
        
        # Check architecture modifications
        if hasattr(metrics, 'expert_utilization'):
            arch_suggestions = self.architecture_evolution.suggest_architecture_changes(
                metrics, self._get_model_info()
            )
            for suggestion in arch_suggestions:
                self._consider_architecture_change(suggestion)
        
        # Predict training trajectory
        trajectory = self.meta_learner.predict_training_trajectory(metrics, self.config)
        if trajectory and trajectory['confidence'] > 0.8:
            self._act_on_trajectory_prediction(trajectory)
    
    def _handle_training_anomaly(self, anomaly):
        """Handle detected training anomalies."""
        if anomaly['type'] == 'gradient_explosion':
            # ✅ Mark as emergency
            adjustment = {
                'factor': 0.1,
                'reasoning': 'EMERGENCY: Gradient explosion detected',
                'emergency': True
            }
            self._apply_learning_rate_adjustment(adjustment)

        elif anomaly['type'] == 'loss_spike':
            # ✅ Mark as emergency if severe
            severity = anomaly.get('severity', 'medium')
            adjustment = {
                'factor': 0.5 if severity == 'critical' else 0.8,
                'reasoning': f'Loss spike detected (severity: {severity})',
                'emergency': severity == 'critical'
            }
            self._apply_learning_rate_adjustment(adjustment)
    
    def _apply_learning_rate_adjustment(self, adjustment):
        """Apply learning rate adjustment - FIXED to bypass scheduler conflicts."""
        if not self.trainer:
            return

        # ✅ CHECK: Is adaptive LR enabled at all?
        if not getattr(self.config, 'enable_adaptive_lr', True):
            if getattr(self.config, 'log_lr_decisions', False):
                logging.info(f"⏸️ Adaptive LR disabled - skipping adjustment: {adjustment['reasoning']}")
            return

        current_lr = getattr(self.trainer, 'current_lr', self.config.learning_rate)
        new_lr = current_lr * adjustment['factor']

        is_emergency = adjustment.get('emergency', False)

        if is_emergency:
            # Emergency changes always apply immediately
            logging.warning(f"🚨 EMERGENCY LR Override")
            logging.warning(f"   Reason: {adjustment['reasoning']}")
            logging.warning(f"   Current LR: {current_lr:.2e} → New LR: {new_lr:.2e}")
        else:
            # Check if change is significant enough
            min_threshold = getattr(self.config, 'min_override_threshold', 0.1)  # ✅ Lowered from 0.2
            change_ratio = abs(new_lr - current_lr) / current_lr

            if change_ratio < min_threshold:
                if getattr(self.config, 'log_lr_decisions', False):
                    logging.info(f"⏸️ LR change too small ({change_ratio:.1%} < {min_threshold:.1%})")
                return

            if getattr(self.config, 'log_lr_decisions', False):
                logging.info(f"📊 Adaptive LR Adjustment")
                logging.info(f"   Reason: {adjustment['reasoning']}")
                logging.info(f"   Current LR: {current_lr:.2e} → New LR: {new_lr:.2e} ({change_ratio:.1%} change)")

        # Create and execute decision
        decision = AdaptiveDecision(
            decision_type='learning_rate_adjustment',
            parameters={
                'old_lr': current_lr, 
                'new_lr': new_lr, 
                'factor': adjustment['factor'],
                'emergency': is_emergency
            },
            confidence=0.9 if is_emergency else 0.7,
            reasoning=adjustment['reasoning'],
            expected_improvement=0.1,
            timestamp=datetime.now()
        )

        self._execute_adaptive_decision(decision)

        # ✅ Update trainer's learning rate
        if hasattr(self.trainer, 'adjust_learning_rate'):
            # Pass emergency flag to set appropriate grace period
            grace_period = 20 if is_emergency else 10
            self.trainer.adjust_learning_rate(new_lr, grace_period=grace_period)
    
    def _execute_adaptive_decision(self, decision):
        """Execute an adaptive decision."""
        self.adaptive_decisions.append(decision)
    
        logging.info(f"Executing adaptive decision: {decision.decision_type}")
        logging.info(f"Reasoning: {decision.reasoning}")
        logging.info(f"Confidence: {decision.confidence:.2f}")

        try:
            # ✅ FIX: Actually handle learning_rate_adjustment!
            if decision.decision_type == 'learning_rate_adjustment':
                new_lr = decision.parameters['new_lr']
                is_emergency = decision.parameters.get('emergency', False)

                if hasattr(self.trainer, 'adjust_learning_rate'):
                    grace_period = 20 if is_emergency else 10
                    self.trainer.adjust_learning_rate(
                        new_lr, 
                        grace_period=grace_period,
                        emergency=is_emergency
                    )
                    logging.info(f"✅ LR adjusted: {new_lr:.2e} (emergency: {is_emergency})")
                else:
                    logging.error("Trainer missing adjust_learning_rate method!")

            elif decision.decision_type == 'emergency_lr_reduction':
                if hasattr(self.trainer, 'emergency_lr_reduction'):
                    self.trainer.emergency_lr_reduction(decision.parameters['factor'])

            elif decision.decision_type == 'checkpoint_rollback':
                if hasattr(self.trainer, 'rollback_steps'):
                    self.trainer.rollback_steps(decision.parameters['steps_back'])

            elif decision.decision_type == 'add_expert':
                if hasattr(self.trainer, 'add_expert'):
                    self.trainer.add_expert()

            elif decision.decision_type == 'prune_expert':
                if hasattr(self.trainer, 'prune_expert'):
                    self.trainer.prune_expert(decision.parameters['expert_id'])

            logging.info(f"Successfully executed: {decision.decision_type}")

        except Exception as e:
            logging.error(f"Failed to execute adaptive decision {decision.decision_type}: {e}")
    
    def initialize_training(self):
        """Initialize training with adaptive intelligence."""
        logging.info("Initializing adaptive training system...")
        
        # Get suggestions from meta-learner
        if len(self.meta_learner.training_history) > 0:
            initial_metrics = TrainingMetrics(
                epoch=0, step=0, loss=float('inf'), grad_norm=0,
                learning_rate=self.config.learning_rate,
                expert_utilization={}, memory_usage={},
                throughput=0, semantic_coherence=0,
                factual_accuracy=0, reasoning_score=0,
                timestamp=datetime.now()
            )
            
            suggestions = self.meta_learner.suggest_hyperparameters(initial_metrics, self.config)
            if suggestions:
                logging.info(f"Meta-learner suggestions: {suggestions}")
                self._apply_meta_suggestions(suggestions)
        
        # Initialize components
        try:
            from core.tokenizer import ConversationTokenizer
            self.tokenizer = ConversationTokenizer()
            if hasattr(self.config, 'vocab_size'):
                self.config.vocab_size = self.tokenizer.vocab_size
            
            from core.model import DeepSeekTransformer
            
            model_config = self.config
            try:
                from Main import config_to_deepseek_config
                model_config = config_to_deepseek_config(self.config)
                logging.info("Using converted DeepSeek config")
            except ImportError:
                logging.info("Using config directly for model initialization")
            
            self.model = DeepSeekTransformer(model_config)
            logging.info("Model initialized with adaptive architecture support")
            
            # Initialize adaptive trainer
            self._initialize_adaptive_trainer()
            
            # Start monitoring
            self.start_real_time_monitoring()
            
            logging.info("Adaptive training system ready")
            
        except Exception as e:
            logging.error(f"Failed to initialize adaptive training: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def _initialize_adaptive_trainer(self):
        """Initialize trainer with adaptive capabilities - FIXED."""
        logging.info("Attempting to initialize EnhancedConversationTrainer...")
    
        # Try to import the real trainer
        trainer_classes = [
            ('training.trainer', 'EnhancedConversationTrainer'),
            ('trainer', 'EnhancedConversationTrainer'),
        ]

        trainer_initialized = False

        for module_name, class_name in trainer_classes:
            try:
                logging.info(f"Trying to import {class_name} from {module_name}...")
                module = __import__(module_name, fromlist=[class_name])
                trainer_class = getattr(module, class_name)
                
                # Pass ALL required arguments to trainer
                self.trainer = trainer_class(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    config=self.config,
                    logger=self.logger
                )

                # Enhance trainer with adaptive capabilities
                self._enhance_trainer_with_adaptive_features()

                logging.info(f"✅ Real trainer initialized: {class_name}")
                trainer_initialized = True
                break
                
            except (ImportError, AttributeError) as e:
                logging.debug(f"Could not import {class_name} from {module_name}: {e}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error importing {class_name}: {e}")
                continue
        
        # FIX: Use fallback trainer if real one not found
        if not trainer_initialized:
            logging.warning("Could not import EnhancedConversationTrainer, using fallback")
            self.trainer = self._create_adaptive_trainer()
            logging.info("✅ Fallback trainer initialized")
    
    def _enhance_trainer_with_adaptive_features(self):
        """Add adaptive features to existing trainer."""
        # Inject monitoring callback
        original_training_step = getattr(self.trainer, 'training_step', None)
        
        def adaptive_training_step(*args, **kwargs):
            result = original_training_step(*args, **kwargs) if original_training_step else None
            
            # Extract metrics and queue for real-time processing
            if hasattr(self.trainer, 'get_current_metrics'):
                metrics = self.trainer.get_current_metrics()
                if metrics:
                    self.monitoring_queue.put(metrics)
            
            return result
        
        self.trainer.training_step = adaptive_training_step
    
    def run_adaptive_training(self):
        """Run training with full adaptive intelligence - FIXED to setup scheduler."""
        logging.info("="*80)
        logging.info("ADAPTIVE AI-DRIVEN TRAINING WITH SELF-IMPROVEMENT")
        logging.info("="*80)

        start_time = datetime.now()

        try:
            self.is_training = True

            # Initialize trainer if needed
            if self.trainer is None:
                logging.warning("Trainer was None, initializing now...")
                self.initialize_training()

            if self.trainer is None:
                raise RuntimeError("CRITICAL: Trainer still None after initialization!")

            logging.info(f"✅ Trainer confirmed: {type(self.trainer).__name__}")

            # Setup datasets
            logging.info("Setting up datasets...")
            train_dataset, eval_dataset = self._setup_datasets()

            if train_dataset is None or len(train_dataset) == 0:
                raise RuntimeError("Training dataset is empty or None!")

            logging.info(f"✅ Train dataset: {len(train_dataset)} samples")
            logging.info(f"✅ Eval dataset: {len(eval_dataset)} samples")

            # ✅ MOVE THE SCHEDULER SETUP HERE (before training)
            logging.info("Setting up learning rate scheduler...")

            if type(self.trainer).__name__ == 'AdaptiveTrainer':
                logging.warning("Using fallback trainer - manual scheduler setup required")
                # Calculate total steps manually
                gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
                batches_per_epoch = len(train_dataset) // self.config.batch_size
                steps_per_epoch = batches_per_epoch // gradient_accumulation_steps
                total_steps = steps_per_epoch * self.config.num_epochs

                # Create scheduler manually for fallback trainer
                from torch.optim.lr_scheduler import LambdaLR
                import math

                warmup_ratio = getattr(self.config, 'warmup_ratio', 0.1)
                warmup_steps = int(total_steps * warmup_ratio)

                def lr_lambda(current_step: int):
                    if current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    else:
                        progress = (current_step - warmup_steps) / max(1, (total_steps - warmup_steps))
                        min_lr_ratio = self.config.min_lr / self.config.learning_rate
                        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

                self.trainer.scheduler = LambdaLR(self.trainer.optimizer, lr_lambda)
                logging.info(f"✅ Manual scheduler created: warmup={warmup_steps}, total={total_steps}")
            else:
                # Real trainer - use its built-in setup
                self._setup_trainer_scheduler(train_dataset)

            # Pre-training analysis
            logging.info("Performing pre-training analysis...")
            self._analyze_dataset_characteristics(train_dataset)

            # THE ACTUAL TRAINING CALL
            logging.info("="*80)
            logging.info("STARTING ACTUAL TRAINING LOOP")
            logging.info("="*80)

            self.trainer.train(train_dataset, eval_dataset)

            logging.info("="*80)
            logging.info("TRAINING LOOP COMPLETED")
            logging.info("="*80)

            # Post-training analysis
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()

            final_performance = self._calculate_final_performance()

            self.meta_learner.record_training_outcome(
                self.config, self.training_metrics_history, final_performance
            )

            self._generate_adaptive_insights_report(training_duration, final_performance)
            self._save_meta_learning_state()

            logging.info(f"Adaptive training completed in {training_duration:.1f} seconds")
            logging.info(f"Made {len(self.adaptive_decisions)} adaptive decisions during training")

        except Exception as e:
            logging.error(f"Adaptive training failed: {e}")
            logging.error(traceback.format_exc())
            self._save_emergency_adaptive_state()
            raise
        finally:
            self.is_training = False
    
    
    def _analyze_dataset_characteristics(self, dataset):
        """Analyze dataset to inform adaptive strategies."""
        try:
            sample_size = min(100, len(dataset))
            sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
            
            token_lengths = []
            for idx in sample_indices:
                sample = dataset[idx]
                if hasattr(sample, 'input_ids'):
                    token_lengths.append(len(sample.input_ids))
            
            if token_lengths:
                characteristics = {
                    'avg_length': np.mean(token_lengths),
                    'std_length': np.std(token_lengths),
                    'max_length': np.max(token_lengths),
                    'min_length': np.min(token_lengths)
                }
                
                logging.info(f"Dataset characteristics: {characteristics}")
                
                # Adjust config based on characteristics
                if characteristics['avg_length'] > self.config.seq_length * 0.8:
                    logging.warning("Dataset has long sequences, consider increasing seq_length")
                
                return characteristics
        except Exception as e:
            logging.warning(f"Could not analyze dataset characteristics: {e}")
        
        return {}
    
    def _calculate_final_performance(self):
        """Calculate final performance metrics."""
        if not self.training_metrics_history:
            return 0.0
        
        recent_metrics = self.training_metrics_history[-10:]
        avg_loss = np.mean([m.loss for m in recent_metrics])
        
        # Normalize performance (lower loss = higher performance)
        performance = max(0, 1.0 - min(avg_loss / 10.0, 1.0))
        return performance
    
    def _generate_adaptive_insights_report(self, training_duration, final_performance):
        """Generate comprehensive report of adaptive insights."""
        report = {
            'experiment_name': self.config.experiment_name,
            'training_duration_seconds': training_duration,
            'final_performance': final_performance,
            'total_adaptive_decisions': len(self.adaptive_decisions),
            'metrics_collected': len(self.training_metrics_history),
            'timestamp': datetime.now().isoformat()
        }
        
        # Categorize decisions
        decision_types = {}
        for decision in self.adaptive_decisions:
            decision_type = decision.decision_type
            if decision_type not in decision_types:
                decision_types[decision_type] = []
            decision_types[decision_type].append(decision.confidence)
        
        report['decision_breakdown'] = {}
        for decision_type, confidences in decision_types.items():
            report['decision_breakdown'][decision_type] = {
                'count': len(confidences),
                'avg_confidence': np.mean(confidences),
                'success_rate': len([c for c in confidences if c > 0.7]) / len(confidences)
            }
        
        # Performance trends
        if len(self.training_metrics_history) > 10:
            losses = [m.loss for m in self.training_metrics_history]
            report['performance_trends'] = {
                'initial_loss': losses[0],
                'final_loss': losses[-1],
                'best_loss': min(losses),
                'convergence_rate': self._calculate_convergence_rate(losses),
                'stability_score': 1.0 - np.std(losses[-20:]) if len(losses) > 20 else 0.5
            }
        
        # Meta-learning insights
        if len(self.meta_learner.training_history) > 1:
            report['meta_learning'] = {
                'historical_runs': len(self.meta_learner.training_history),
                'improvement_over_baseline': final_performance - 0.5,
                'learned_strategies': len(self.meta_learner.successful_strategies)
            }
        
        # Save report
        report_path = self.experiment_dir / "adaptive_insights_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log key insights
        logging.info("\n" + "="*60)
        logging.info("ADAPTIVE TRAINING INSIGHTS")
        logging.info("="*60)
        logging.info(f"Final Performance: {final_performance:.3f}")
        logging.info(f"Adaptive Decisions Made: {len(self.adaptive_decisions)}")
        
        for decision_type, stats in report['decision_breakdown'].items():
            logging.info(f"{decision_type}: {stats['count']} decisions, "
                        f"{stats['avg_confidence']:.2f} avg confidence, "
                        f"{stats['success_rate']:.2%} success rate")
        
        if 'performance_trends' in report:
            trends = report['performance_trends']
            improvement = trends['initial_loss'] - trends['final_loss']
            logging.info(f"Loss Improvement: {improvement:.3f} "
                        f"({improvement/trends['initial_loss']:.1%} reduction)")
        
        logging.info("="*60)
    
    def _calculate_convergence_rate(self, losses):
        """Calculate how quickly the model converged."""
        if len(losses) < 10:
            return 0.0
        
        # Fit exponential decay to estimate convergence
        steps = np.arange(len(losses))
        try:
            # Simple linear fit to log losses (exponential decay)
            log_losses = np.log(np.array(losses) + 1e-8)
            coeffs = np.polyfit(steps, log_losses, 1)
            return abs(coeffs[0])
        except:
            return 0.0
    
    def _apply_meta_suggestions(self, suggestions):
        """Apply suggestions from meta-learner."""
        for suggestion_type, params in suggestions.items():
            if suggestion_type == 'learning_rate' and 'value' in params:
                old_lr = self.config.learning_rate
                self.config.learning_rate = params['value']
                logging.info(f"Meta-learner adjusted learning rate: {old_lr} -> {params['value']}")
            
            elif suggestion_type == 'batch_size' and 'value' in params:
                old_batch = self.config.batch_size
                self.config.batch_size = params['value']
                logging.info(f"Meta-learner adjusted batch size: {old_batch} -> {params['value']}")
    
    def _get_model_info(self):
        """Get current model information."""
        if not self.model:
            return {}
        
        info = {
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Add expert information if available
        if hasattr(self.model, 'num_experts'):
            info['num_experts'] = self.model.num_experts
        
        return info
    
    def _act_on_loss_insights(self, insights):
        """Act on loss curve insights."""
        if insights['trend_direction'] == 'increasing' and insights['trend_strength'] > 0.01:
            # Loss is increasing, take corrective action
            decision = AdaptiveDecision(
                decision_type='corrective_lr_reduction',
                parameters={'factor': 0.8, 'reason': 'increasing_loss_trend'},
                confidence=0.6,
                reasoning=f"Loss trend increasing with strength {insights['trend_strength']:.4f}",
                expected_improvement=0.2,
                timestamp=datetime.now()
            )
            self._execute_adaptive_decision(decision)
        
        elif insights['curvature'] == 'concave_up' and insights['trend_direction'] == 'decreasing':
            # Good convergence, might benefit from slight LR increase
            decision = AdaptiveDecision(
                decision_type='optimization_lr_increase',
                parameters={'factor': 1.1, 'reason': 'healthy_convergence'},
                confidence=0.5,
                reasoning="Healthy convergence detected, slight LR increase might help",
                expected_improvement=0.05,
                timestamp=datetime.now()
            )
            if decision.confidence > 0.7:
                self._execute_adaptive_decision(decision)
    
    def _act_on_trajectory_prediction(self, trajectory):
        """Act on training trajectory predictions."""
        if trajectory['prediction'] == 'plateau' and trajectory['confidence'] > 0.8:
            # Training is plateauing
            decision = AdaptiveDecision(
                decision_type='plateau_intervention',
                parameters={'action': trajectory['suggested_action']},
                confidence=trajectory['confidence'],
                reasoning=f"Predicted plateau with {trajectory['confidence']:.1%} confidence",
                expected_improvement=trajectory['expected_improvement'],
                timestamp=datetime.now()
            )
            self._execute_adaptive_decision(decision)
        
        elif trajectory['prediction'] == 'potential_divergence':
            # Training might diverge
            decision = AdaptiveDecision(
                decision_type='divergence_prevention',
                parameters={'action': 'emergency_lr_reduction', 'factor': 0.5},
                confidence=trajectory['confidence'],
                reasoning="Potential divergence detected",
                expected_improvement=trajectory['expected_improvement'],
                timestamp=datetime.now()
            )
            if decision.confidence > 0.8:
                self._execute_adaptive_decision(decision)

    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get comprehensive scheduler status for debugging.
        
        Returns:
            Dictionary with scheduler state information
        """
        if not self.trainer or not hasattr(self.trainer, 'scheduler'):
            return {
                'status': 'No trainer or scheduler',
                'has_trainer': self.trainer is not None,
                'trainer_type': type(self.trainer).__name__ if self.trainer else None
            }
        
        scheduler = self.trainer.scheduler
        if scheduler is None:
            return {
                'status': 'Scheduler is None',
                'trainer_has_scheduler_attr': True,
                'scheduler_value': None
            }
        
        try:
            status = {
                'status': 'Active',
                'scheduler_type': type(scheduler).__name__,
                'current_lr': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 'Unknown',
                'base_lrs': scheduler.base_lrs if hasattr(scheduler, 'base_lrs') else 'Unknown',
                'last_epoch': scheduler.last_epoch if hasattr(scheduler, 'last_epoch') else 'Unknown',
                'global_step': self.global_step,
                'trainer_current_lr': getattr(self.trainer, 'current_lr', 'Unknown'),
                'config_lr': self.config.learning_rate,
            }
            
            # Add scheduler-specific info
            if hasattr(scheduler, 'T_max'):
                status['cosine_T_max'] = scheduler.T_max
            if hasattr(scheduler, 'eta_min'):
                status['cosine_eta_min'] = scheduler.eta_min
                
            return status
            
        except Exception as e:
            return {
                'status': 'Error reading scheduler',
                'error': str(e),
                'scheduler_type': type(scheduler).__name__
            }
    
    def _consider_architecture_change(self, suggestion):
        """Consider and potentially apply architecture changes."""
        confidence_threshold = 0.8
        
        if suggestion['action'] == 'add_expert' and len(self.adaptive_decisions) < 10:
            # Only add experts early in training and sparingly
            decision = AdaptiveDecision(
                decision_type='add_expert',
                parameters=suggestion,
                confidence=0.7,
                reasoning=suggestion['reasoning'],
                expected_improvement=suggestion['expected_improvement'],
                timestamp=datetime.now()
            )
            
            if decision.confidence > confidence_threshold:
                self._execute_adaptive_decision(decision)
    
    def _setup_datasets(self):
        """Setup datasets with adaptive loading strategies - FIXED."""
        logging.info("Setting up datasets with adaptive loading...")

        # FIX: Try multiple import paths
        try:
            from core.dataset import setup_datasets
            train_dataset, eval_dataset = setup_datasets(self.config, self.tokenizer)
            logging.info(f"Train dataset ready: {len(train_dataset):,} samples")
            if eval_dataset != train_dataset:
                logging.info(f"Eval dataset ready: {len(eval_dataset):,} samples")
            else:
                logging.info("Using training dataset for evaluation")
            return train_dataset, eval_dataset
        except ImportError:
            pass
        
        try:
            from dataset import setup_datasets
            train_dataset, eval_dataset = setup_datasets(self.config, self.tokenizer)
            logging.info(f"Train dataset ready: {len(train_dataset):,} samples")
            if eval_dataset != train_dataset:
                logging.info(f"Eval dataset ready: {len(eval_dataset):,} samples")
            return train_dataset, eval_dataset
        except ImportError:
            raise ImportError("Could not import dataset setup functions from core.dataset or dataset")
    
    def _save_emergency_adaptive_state(self):
        """Save emergency state including adaptive decisions."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            emergency_path = self.experiment_dir / f"emergency_adaptive_state_{timestamp}.json"
            
            state = {
                'adaptive_decisions': [
                    {
                        'decision_type': d.decision_type,
                        'parameters': d.parameters,
                        'confidence': d.confidence,
                        'reasoning': d.reasoning,
                        'expected_improvement': d.expected_improvement,
                        'timestamp': d.timestamp.isoformat()
                    }
                    for d in self.adaptive_decisions
                ],
                'metrics_history_count': len(self.training_metrics_history),
                'meta_learning_runs': len(self.meta_learner.training_history),
                'experiment_name': self.config.experiment_name,
                'timestamp': timestamp
            }
            
            with open(emergency_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logging.info(f"Emergency adaptive state saved: {emergency_path}")
        except Exception as e:
            logging.error(f"Failed to save emergency adaptive state: {e}")
    
    def _create_adaptive_trainer(self):
        """Create adaptive trainer as fallback."""
        class AdaptiveTrainer:
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
                self.current_lr = config.learning_rate
                self.scheduler = None
            
            def _setup_scheduler(self, total_steps):
                """Setup scheduler (placeholder)."""
                logging.info(f"Fallback trainer: scheduler setup called with {total_steps} steps")

                # ✅ FIX: Actually create a scheduler instead of leaving it None
                from torch.optim.lr_scheduler import LambdaLR
                import math

                warmup_ratio = getattr(self.config, 'warmup_ratio', 0.1)
                warmup_steps = int(total_steps * warmup_ratio)

                def lr_lambda(current_step: int):
                    if current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    else:
                        progress = (current_step - warmup_steps) / max(1, (total_steps - warmup_steps))
                        return max(0.0, 1.0 - progress)

                self.scheduler = LambdaLR(self.optimizer, lr_lambda)
                logging.info(f"✅ Fallback scheduler initialized: warmup={warmup_steps}, total={total_steps}")
            
            def train(self, train_dataset, eval_dataset=None):
                logging.info("Using adaptive fallback trainer")
                for epoch in range(self.config.num_epochs):
                    if self.should_stop:
                        break
                    
                    self.current_epoch = epoch
                    time.sleep(1)
                    
                    # Generate mock metrics
                    mock_metrics = TrainingMetrics(
                        epoch=epoch,
                        step=self.global_step,
                        loss=max(0.1, 5.0 * np.exp(-epoch * 0.1)),
                        grad_norm=np.random.uniform(0.1, 2.0),
                        learning_rate=self.current_lr,
                        expert_utilization={f'expert_{i}': np.random.random() for i in range(4)},
                        memory_usage={'gpu_memory_percent': np.random.uniform(60, 85)},
                        throughput=np.random.uniform(100, 200),
                        semantic_coherence=np.random.uniform(0.7, 0.9),
                        factual_accuracy=np.random.uniform(0.6, 0.8),
                        reasoning_score=np.random.uniform(0.5, 0.8),
                        timestamp=datetime.now()
                    )
                    
                    if hasattr(self, '_orchestrator_queue'):
                        self._orchestrator_queue.put(mock_metrics)
                    
                    self.global_step += 1
            
            def get_current_metrics(self):
                return TrainingMetrics(
                    epoch=self.current_epoch,
                    step=self.global_step,
                    loss=np.random.uniform(0.1, 2.0),
                    grad_norm=np.random.uniform(0.1, 5.0),
                    learning_rate=self.current_lr,
                    expert_utilization={f'expert_{i}': np.random.random() for i in range(4)},
                    memory_usage={'gpu_memory_percent': np.random.uniform(60, 90)},
                    throughput=np.random.uniform(100, 300),
                    semantic_coherence=np.random.uniform(0.6, 0.9),
                    factual_accuracy=np.random.uniform(0.5, 0.8),
                    reasoning_score=np.random.uniform(0.4, 0.8),
                    timestamp=datetime.now()
                )
            
            def adjust_learning_rate(self, new_lr):
                """Adjust learning rate and signal to skip scheduler."""
                self.current_lr = new_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

                # ✅ Signal that adaptive has control
                self._adaptive_lr_override = True
                self._adaptive_override_steps = 0

                logging.info(f"Learning rate adjusted to {new_lr}")
        
        trainer = AdaptiveTrainer(self.model, self.tokenizer, self.config, self.logger)
        trainer._orchestrator_queue = self.monitoring_queue
        return trainer
    
    def get_adaptive_status(self):
        """Get comprehensive adaptive training status."""
        status = {
            'is_training': self.is_training,
            'should_stop': self.should_stop,
            'experiment_name': self.config.experiment_name,
            'experiment_dir': str(self.experiment_dir),
            'adaptive_decisions_made': len(self.adaptive_decisions),
            'metrics_collected': len(self.training_metrics_history),
            'meta_learning_runs': len(self.meta_learner.training_history),
            'monitoring_active': self.monitoring_thread and self.monitoring_thread.is_alive()
        }
        
        if self.current_metrics:
            status['current_metrics'] = self.current_metrics.to_dict()
        
        if self.trainer:
            trainer_status = {}
            for attr in ['current_epoch', 'global_step', 'best_eval_loss', 'patience_counter']:
                trainer_status[attr] = getattr(self.trainer, attr, None)
            status.update(trainer_status)
        
        if self.adaptive_decisions:
            recent_decisions = self.adaptive_decisions[-5:]
            status['recent_decisions'] = [
                {
                    'type': d.decision_type,
                    'confidence': d.confidence,
                    'reasoning': d.reasoning[:100] + '...' if len(d.reasoning) > 100 else d.reasoning,
                    'timestamp': d.timestamp.isoformat()
                }
                for d in recent_decisions
            ]
        
        return status
    
    def cleanup(self):
        """Clean up adaptive training resources."""
        logging.info("Cleaning up adaptive training system...")
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.should_stop = True
            self.monitoring_thread.join(timeout=5)
        
        # Save final adaptive state
        self._save_meta_learning_state()
        
        # Standard cleanup
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
        
        logging.info("Adaptive training orchestrator cleanup completed")


def create_adaptive_orchestrator(config):
    """Factory function to create an AdaptiveTrainingOrchestrator."""
    return AdaptiveTrainingOrchestrator(config)


# Backwards compatibility
TrainingOrchestrator = AdaptiveTrainingOrchestrator