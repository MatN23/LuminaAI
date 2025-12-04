# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import torch
import math
from typing import Optional, Callable, List, Dict
from torch.optim.lr_scheduler import _LRScheduler


class ZeROLRScheduler(_LRScheduler):
    """
    Base learning rate scheduler for ZeRO-compatible optimizers.
    Handles partitioned optimizers and expert-specific learning rates.
    """
    
    def __init__(
        self,
        optimizer,
        last_epoch: int = -1,
        verbose: bool = False,
        expert_registry: Optional[Dict] = None,
    ):
        self.expert_registry = expert_registry or {}
        # FIX: Pass verbose as keyword argument, not positional
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        """Override this method in subclasses"""
        raise NotImplementedError


class CosineAnnealingWarmRestarts(ZeROLRScheduler):
    """
    Cosine annealing with warm restarts for ZeRO optimizers.
    Implements SGDR (Stochastic Gradient Descent with Warm Restarts).
    """
    
    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
        expert_registry: Optional[Dict] = None,
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.T_i = T_0
        
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * (
                1 + math.cos(math.pi * self.T_cur / self.T_i)
            ) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class LinearWarmupCosineDecay(ZeROLRScheduler):
    """
    Linear warmup followed by cosine decay.
    Standard schedule for transformer training.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
        expert_registry: Optional[Dict] = None,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = float(self.last_epoch - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class InverseSquareRootSchedule(ZeROLRScheduler):
    """
    Inverse square root learning rate decay.
    Used in original Transformer paper.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
        expert_registry: Optional[Dict] = None,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        step = max(1, self.last_epoch)
        
        if step < self.warmup_steps:
            # Linear warmup
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Inverse square root decay
            decay_factor = (self.warmup_steps ** 0.5) / (step ** 0.5)
            return [base_lr * decay_factor for base_lr in self.base_lrs]


class PolynomialDecay(ZeROLRScheduler):
    """
    Polynomial learning rate decay.
    """
    
    def __init__(
        self,
        optimizer,
        max_steps: int,
        end_lr: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False,
        expert_registry: Optional[Dict] = None,
    ):
        self.max_steps = max_steps
        self.end_lr = end_lr
        self.power = power
        
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        if self.last_epoch >= self.max_steps:
            return [self.end_lr for _ in self.base_lrs]
        
        decay_factor = (1 - self.last_epoch / self.max_steps) ** self.power
        
        return [
            self.end_lr + (base_lr - self.end_lr) * decay_factor
            for base_lr in self.base_lrs
        ]


class ExponentialDecay(ZeROLRScheduler):
    """
    Exponential learning rate decay.
    """
    
    def __init__(
        self,
        optimizer,
        decay_rate: float,
        decay_steps: int,
        staircase: bool = False,
        last_epoch: int = -1,
        verbose: bool = False,
        expert_registry: Optional[Dict] = None,
    ):
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.staircase = staircase
        
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        if self.staircase:
            decay_factor = self.decay_rate ** (self.last_epoch // self.decay_steps)
        else:
            decay_factor = self.decay_rate ** (self.last_epoch / self.decay_steps)
        
        return [base_lr * decay_factor for base_lr in self.base_lrs]


class OneCycleLR(ZeROLRScheduler):
    """
    One-cycle learning rate policy.
    Gradually increases LR then decreases it, useful for fast convergence.
    """
    
    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
        verbose: bool = False,
        expert_registry: Optional[Dict] = None,
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
        
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.step_size_up:
            # Ascending phase
            pct = step / self.step_size_up
            if self.anneal_strategy == 'cos':
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * (
                    1 - math.cos(math.pi * pct)
                ) / 2
            else:  # linear
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * pct
        else:
            # Descending phase
            pct = (step - self.step_size_up) / self.step_size_down
            if self.anneal_strategy == 'cos':
                lr = self.min_lr + (self.max_lr - self.min_lr) * (
                    1 + math.cos(math.pi * pct)
                ) / 2
            else:  # linear
                lr = self.max_lr - (self.max_lr - self.min_lr) * pct
        
        return [lr for _ in self.base_lrs]


class ExpertAwareLRScheduler(ZeROLRScheduler):
    """
    Learning rate scheduler with expert-specific schedules.
    Useful for MoE/MoD where different experts may need different learning rates.
    """
    
    def __init__(
        self,
        optimizer,
        base_schedule: str = 'cosine',
        expert_lr_multipliers: Optional[Dict[str, float]] = None,
        warmup_steps: int = 0,
        max_steps: int = 1000,
        last_epoch: int = -1,
        verbose: bool = False,
        expert_registry: Optional[Dict] = None,
        **schedule_kwargs,
    ):
        self.base_schedule = base_schedule
        self.expert_lr_multipliers = expert_lr_multipliers or {}
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.schedule_kwargs = schedule_kwargs
        
        # Create base scheduler (will be initialized after super().__init__)
        self.base_scheduler = None
        
        super().__init__(optimizer, last_epoch=last_epoch)
        
        # Now create base scheduler after parent initialization
        if base_schedule == 'cosine':
            self.base_scheduler = LinearWarmupCosineDecay(
                optimizer, warmup_steps, max_steps, last_epoch=last_epoch, **schedule_kwargs
            )
        elif base_schedule == 'inverse_sqrt':
            self.base_scheduler = InverseSquareRootSchedule(
                optimizer, warmup_steps, last_epoch=last_epoch, **schedule_kwargs
            )
        elif base_schedule == 'polynomial':
            self.base_scheduler = PolynomialDecay(
                optimizer, max_steps, last_epoch=last_epoch, **schedule_kwargs
            )
        else:
            raise ValueError(f"Unknown schedule: {base_schedule}")
        
        # Build parameter to expert mapping
        self.param_to_expert: Dict[int, str] = {}
        self._build_param_expert_map()
    
    def _build_param_expert_map(self):
        """Map parameters to their expert owners"""
        for expert_name, expert_module in self.expert_registry.items():
            for param in expert_module.parameters():
                self.param_to_expert[id(param)] = expert_name
    
    def get_lr(self):
        """Get learning rates with expert-specific multipliers"""
        if self.base_scheduler is None:
            return self.base_lrs
            
        base_lrs = self.base_scheduler.get_lr()
        
        # Apply expert multipliers
        adjusted_lrs = []
        for i, (param_group, base_lr) in enumerate(zip(self.optimizer.param_groups, base_lrs)):
            # Check if this param group contains expert parameters
            expert_multiplier = 1.0
            for param in param_group['params']:
                param_id = id(param)
                expert_name = self.param_to_expert.get(param_id)
                if expert_name and expert_name in self.expert_lr_multipliers:
                    expert_multiplier = self.expert_lr_multipliers[expert_name]
                    break
            
            adjusted_lrs.append(base_lr * expert_multiplier)
        
        return adjusted_lrs
    
    def step(self, epoch=None):
        """Step both base scheduler and apply expert multipliers"""
        if self.base_scheduler:
            self.base_scheduler.step(epoch)
        super().step(epoch)


class WarmupThenConstant(ZeROLRScheduler):
    """
    Linear warmup then constant learning rate.
    Simple but effective for many tasks.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
        expert_registry: Optional[Dict] = None,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


def create_scheduler(
    optimizer,
    scheduler_name: str,
    warmup_steps: int = 0,
    max_steps: int = 1000,
    expert_registry: Optional[Dict] = None,
    expert_lr_multipliers: Optional[Dict[str, float]] = None,
    **kwargs,
) -> ZeROLRScheduler:
    """
    Factory function to create learning rate schedulers.
    
    Args:
        optimizer: ZeRO-compatible optimizer
        scheduler_name: Name of scheduler ('cosine', 'inverse_sqrt', 'polynomial', 
                       'exponential', 'one_cycle', 'expert_aware', 'warmup_constant')
        warmup_steps: Number of warmup steps
        max_steps: Maximum training steps
        expert_registry: LuminaAI expert registry
        expert_lr_multipliers: Expert-specific learning rate multipliers
        **kwargs: Additional scheduler-specific arguments
    
    Returns:
        Learning rate scheduler
    """
    scheduler_map = {
        'cosine': LinearWarmupCosineDecay,
        'cosine_restarts': CosineAnnealingWarmRestarts,
        'inverse_sqrt': InverseSquareRootSchedule,
        'polynomial': PolynomialDecay,
        'exponential': ExponentialDecay,
        'one_cycle': OneCycleLR,
        'expert_aware': ExpertAwareLRScheduler,
        'warmup_constant': WarmupThenConstant,
    }
    
    if scheduler_name.lower() not in scheduler_map:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    scheduler_class = scheduler_map[scheduler_name.lower()]
    
    # Build kwargs based on scheduler type
    if scheduler_name.lower() in ['cosine', 'polynomial']:
        return scheduler_class(
            optimizer, 
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            expert_registry=expert_registry,
            **kwargs
        )
    elif scheduler_name.lower() == 'inverse_sqrt':
        return scheduler_class(
            optimizer,
            warmup_steps=warmup_steps,
            expert_registry=expert_registry,
            **kwargs
        )
    elif scheduler_name.lower() == 'expert_aware':
        return scheduler_class(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            expert_registry=expert_registry,
            expert_lr_multipliers=expert_lr_multipliers,
            **kwargs
        )
    else:
        return scheduler_class(
            optimizer,
            expert_registry=expert_registry,
            **kwargs
        )