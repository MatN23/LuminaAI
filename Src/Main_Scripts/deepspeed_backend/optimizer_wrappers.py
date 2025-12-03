"""
ZeRO-Compatible Optimizer Wrappers for LuminaAI
Wraps standard and 8-bit optimizers for ZeRO stage compliance.
"""

import torch
import torch.optim as optim
from typing import Dict, List, Optional, Any, Callable, Iterable
from collections import defaultdict


class ZeROOptimizer:
    """
    Base wrapper for ZeRO-compatible optimizers.
    Handles state partitioning, gradient clipping, and expert-aware updates.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        zero_stage: int = 0,
        partition_rank: int = 0,
        world_size: int = 1,
        expert_registry: Optional[Dict] = None,
        gradient_clipping: Optional[float] = None,
        offload_manager: Optional[Any] = None,
    ):
        self.optimizer = optimizer
        self.zero_stage = zero_stage
        self.partition_rank = partition_rank
        self.world_size = world_size
        self.expert_registry = expert_registry or {}
        self.gradient_clipping = gradient_clipping
        self.offload_manager = offload_manager
        
        # Track which parameters this rank owns
        self.owned_params: List[torch.nn.Parameter] = []
        self._partition_parameters()
        
        # Step tracking
        self.step_count = 0
        
        # Expert-specific learning rates
        self.expert_lr_scales: Dict[str, float] = {}
    
    def _partition_parameters(self):
        """Determine which parameters this rank owns based on ZeRO stage"""
        if self.zero_stage == 0:
            # No partitioning, all ranks own all params
            self.owned_params = list(self.optimizer.param_groups[0]['params'])
            return
        
        all_params = []
        for param_group in self.optimizer.param_groups:
            all_params.extend(param_group['params'])
        
        # Partition based on hash
        for param in all_params:
            param_id = id(param)
            if param_id % self.world_size == self.partition_rank:
                self.owned_params.append(param)
    
    def step(self, closure: Optional[Callable] = None):
        """Execute optimizer step with ZeRO awareness"""
        # Gradient clipping
        if self.gradient_clipping is not None:
            self._clip_gradients()
        
        # Offload optimizer states if configured
        if self.offload_manager and self.zero_stage >= 1:
            self._offload_states_before_step()
        
        # Update only owned parameters
        if self.zero_stage >= 1:
            self._step_partitioned(closure)
        else:
            self.optimizer.step(closure)
        
        # Restore optimizer states if offloaded
        if self.offload_manager and self.zero_stage >= 1:
            self._restore_states_after_step()
        
        self.step_count += 1
    
    def _step_partitioned(self, closure: Optional[Callable] = None):
        """Step only owned parameters"""
        # Temporarily zero out gradients for non-owned params
        original_grads = {}
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param not in self.owned_params and param.grad is not None:
                    original_grads[param] = param.grad
                    param.grad = None
        
        # Step optimizer
        self.optimizer.step(closure)
        
        # Restore gradients
        for param, grad in original_grads.items():
            param.grad = grad
    
    def _clip_gradients(self):
        """Apply gradient clipping"""
        params_with_grad = [p for p in self.owned_params if p.grad is not None]
        if params_with_grad:
            torch.nn.utils.clip_grad_norm_(params_with_grad, self.gradient_clipping)
    
    def _offload_states_before_step(self):
        """Offload optimizer states before step"""
        for param in self.owned_params:
            param_id = id(param)
            if param_id in self.optimizer.state:
                state = self.optimizer.state[param_id]
                self.offload_manager.offload_optimizer_state(param_id, state)
    
    def _restore_states_after_step(self):
        """Restore optimizer states after step"""
        device = self.owned_params[0].device if self.owned_params else 'cuda'
        for param in self.owned_params:
            param_id = id(param)
            restored_state = self.offload_manager.restore_optimizer_state(param_id, device)
            if restored_state:
                self.optimizer.state[param_id] = restored_state
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients"""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> Dict:
        """Return optimizer state dict"""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state dict"""
        self.optimizer.load_state_dict(state_dict)
    
    def set_expert_lr_scale(self, expert_name: str, scale: float):
        """
        Set learning rate scale for specific expert (useful for MoE).
        
        Args:
            expert_name: Name of expert
            scale: Learning rate multiplier
        """
        self.expert_lr_scales[expert_name] = scale
    
    @property
    def param_groups(self):
        """Expose param_groups for compatibility"""
        return self.optimizer.param_groups


class ZeROAdamW(ZeROOptimizer):
    """ZeRO-compatible AdamW optimizer wrapper"""
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        zero_stage: int = 0,
        partition_rank: int = 0,
        world_size: int = 1,
        expert_registry: Optional[Dict] = None,
        gradient_clipping: Optional[float] = None,
        offload_manager: Optional[Any] = None,
    ):
        optimizer = optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        
        super().__init__(
            optimizer=optimizer,
            zero_stage=zero_stage,
            partition_rank=partition_rank,
            world_size=world_size,
            expert_registry=expert_registry,
            gradient_clipping=gradient_clipping,
            offload_manager=offload_manager,
        )


class ZeROAdam8bit(ZeROOptimizer):
    """
    ZeRO-compatible 8-bit Adam optimizer wrapper.
    Reduces memory footprint of optimizer states by 4x.
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        zero_stage: int = 0,
        partition_rank: int = 0,
        world_size: int = 1,
        expert_registry: Optional[Dict] = None,
        gradient_clipping: Optional[float] = None,
        offload_manager: Optional[Any] = None,
        percentile_clipping: int = 100,
        block_wise: bool = True,
    ):
        # Try to import bitsandbytes for 8-bit optimization
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                percentile_clipping=percentile_clipping,
                block_wise=block_wise,
            )
        except ImportError:
            print("[Warning] bitsandbytes not found, falling back to standard Adam")
            optimizer = optim.Adam(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        
        super().__init__(
            optimizer=optimizer,
            zero_stage=zero_stage,
            partition_rank=partition_rank,
            world_size=world_size,
            expert_registry=expert_registry,
            gradient_clipping=gradient_clipping,
            offload_manager=offload_manager,
        )
        
        self.is_8bit = 'bitsandbytes' in str(type(optimizer))


class ZeROAdafactor(ZeROOptimizer):
    """
    ZeRO-compatible Adafactor optimizer.
    Memory-efficient optimizer that doesn't store momentum.
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: Optional[float] = None,
        eps: tuple = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
        zero_stage: int = 0,
        partition_rank: int = 0,
        world_size: int = 1,
        expert_registry: Optional[Dict] = None,
        gradient_clipping: Optional[float] = None,
        offload_manager: Optional[Any] = None,
    ):
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                params,
                lr=lr,
                eps=eps,
                clip_threshold=clip_threshold,
                decay_rate=decay_rate,
                beta1=beta1,
                weight_decay=weight_decay,
                scale_parameter=scale_parameter,
                relative_step=relative_step,
                warmup_init=warmup_init,
            )
        except ImportError:
            print("[Warning] transformers not found, falling back to Adam")
            optimizer = optim.Adam(params, lr=lr or 1e-3)
        
        super().__init__(
            optimizer=optimizer,
            zero_stage=zero_stage,
            partition_rank=partition_rank,
            world_size=world_size,
            expert_registry=expert_registry,
            gradient_clipping=gradient_clipping,
            offload_manager=offload_manager,
        )


class ZeROLARS(ZeROOptimizer):
    """
    ZeRO-compatible LARS optimizer.
    Useful for large batch training.
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        dampening: float = 0,
        nesterov: bool = False,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
        zero_stage: int = 0,
        partition_rank: int = 0,
        world_size: int = 1,
        expert_registry: Optional[Dict] = None,
        gradient_clipping: Optional[float] = None,
        offload_manager: Optional[Any] = None,
    ):
        # LARS is essentially SGD with layer-wise adaptive learning rates
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
        
        super().__init__(
            optimizer=optimizer,
            zero_stage=zero_stage,
            partition_rank=partition_rank,
            world_size=world_size,
            expert_registry=expert_registry,
            gradient_clipping=gradient_clipping,
            offload_manager=offload_manager,
        )
        
        self.trust_coefficient = trust_coefficient
        self.eps = eps
    
    def step(self, closure: Optional[Callable] = None):
        """LARS step with layer-wise adaptive learning rates"""
        # Apply LARS adaptation
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is None or param not in self.owned_params:
                    continue
                
                param_norm = torch.norm(param.data)
                grad_norm = torch.norm(param.grad.data)
                
                if param_norm > 0 and grad_norm > 0:
                    # Compute adaptive learning rate
                    adaptive_lr = self.trust_coefficient * param_norm / (
                        grad_norm + self.eps
                    )
                    
                    # Scale gradient
                    param.grad.data *= adaptive_lr
        
        # Regular step
        super().step(closure)


class ExpertAwareOptimizer(ZeROOptimizer):
    """
    Optimizer wrapper with expert-specific learning rates and weight decay.
    Useful for MoE/MoD where different experts may need different hyperparameters.
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: str = 'adamw',
        lr: float = 1e-3,
        expert_lr_multipliers: Optional[Dict[str, float]] = None,
        expert_weight_decay: Optional[Dict[str, float]] = None,
        zero_stage: int = 0,
        partition_rank: int = 0,
        world_size: int = 1,
        expert_registry: Optional[Dict] = None,
        gradient_clipping: Optional[float] = None,
        offload_manager: Optional[Any] = None,
        **optimizer_kwargs,
    ):
        # Create base optimizer
        if base_optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(params, lr=lr, **optimizer_kwargs)
        elif base_optimizer.lower() == 'adam':
            optimizer = optim.Adam(params, lr=lr, **optimizer_kwargs)
        elif base_optimizer.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=lr, **optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {base_optimizer}")
        
        super().__init__(
            optimizer=optimizer,
            zero_stage=zero_stage,
            partition_rank=partition_rank,
            world_size=world_size,
            expert_registry=expert_registry,
            gradient_clipping=gradient_clipping,
            offload_manager=offload_manager,
        )
        
        self.expert_lr_multipliers = expert_lr_multipliers or {}
        self.expert_weight_decay = expert_weight_decay or {}
        self.base_lr = lr
        
        # Build parameter to expert mapping
        self.param_to_expert: Dict[int, str] = {}
        self._build_param_expert_map()
    
    def _build_param_expert_map(self):
        """Map parameters to their expert owners"""
        for expert_name, expert_module in self.expert_registry.items():
            for param in expert_module.parameters():
                self.param_to_expert[id(param)] = expert_name
    
    def step(self, closure: Optional[Callable] = None):
        """Step with expert-specific learning rates"""
        # Apply expert-specific learning rates
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param not in self.owned_params:
                    continue
                
                param_id = id(param)
                expert_name = self.param_to_expert.get(param_id)
                
                if expert_name and expert_name in self.expert_lr_multipliers:
                    # Temporarily scale learning rate
                    original_lr = param_group['lr']
                    param_group['lr'] = self.base_lr * self.expert_lr_multipliers[expert_name]
        
        # Regular step
        super().step(closure)
        
        # Restore original learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr


def create_zero_optimizer(
    optimizer_name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float = 1e-3,
    zero_stage: int = 0,
    partition_rank: int = 0,
    world_size: int = 1,
    expert_registry: Optional[Dict] = None,
    gradient_clipping: Optional[float] = None,
    offload_manager: Optional[Any] = None,
    **kwargs,
) -> ZeROOptimizer:
    """
    Factory function to create ZeRO-compatible optimizers.
    
    Args:
        optimizer_name: 'adamw', 'adam8bit', 'adafactor', 'lars', 'expert_aware'
        params: Model parameters
        lr: Learning rate
        zero_stage: ZeRO stage (0, 1, 2, 3)
        partition_rank: Rank in distributed training
        world_size: Total number of ranks
        expert_registry: LuminaAI expert registry
        gradient_clipping: Gradient clipping threshold
        offload_manager: Offload manager instance
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        ZeRO-compatible optimizer
    """
    optimizer_map = {
        'adamw': ZeROAdamW,
        'adam8bit': ZeROAdam8bit,
        'adafactor': ZeROAdafactor,
        'lars': ZeROLARS,
        'expert_aware': ExpertAwareOptimizer,
    }
    
    if optimizer_name.lower() not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    optimizer_class = optimizer_map[optimizer_name.lower()]
    
    return optimizer_class(
        params=params,
        lr=lr,
        zero_stage=zero_stage,
        partition_rank=partition_rank,
        world_size=world_size,
        expert_registry=expert_registry,
        gradient_clipping=gradient_clipping,
        offload_manager=offload_manager,
        **kwargs,
    )