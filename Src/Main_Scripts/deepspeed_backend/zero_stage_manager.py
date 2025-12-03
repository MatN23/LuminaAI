"""
ZeRO Stage Manager for LuminaAI
Implements ZeRO stages 1, 2, 3 with hooks for MoE/MoD expert partitioning.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from enum import Enum


class ZeROStage(Enum):
    """ZeRO optimization stages"""
    DISABLED = 0
    OPTIMIZER_STATES = 1  # ZeRO-1: Partition optimizer states
    GRADIENTS = 2  # ZeRO-2: Partition optimizer states + gradients
    PARAMETERS = 3  # ZeRO-3: Partition optimizer states + gradients + parameters


class ZeROStageManager:
    """
    Manages ZeRO optimization stages with LuminaAI MoE/MoD expert awareness.
    
    Args:
        stage: ZeRO stage (0, 1, 2, or 3)
        model: PyTorch model
        optimizer: Optimizer instance
        expert_registry: Optional dict mapping expert names to modules
        overlap_comm: Enable communication/computation overlap
        cpu_offload: Offload to CPU memory
        nvme_offload: Offload to NVMe storage
    """
    
    def __init__(
        self,
        stage: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        expert_registry: Optional[Dict[str, torch.nn.Module]] = None,
        overlap_comm: bool = True,
        cpu_offload: bool = False,
        nvme_offload: bool = False,
        partition_size: int = 1e9,  # 1GB default partition size
    ):
        self.stage = ZeROStage(stage)
        self.model = model
        self.optimizer = optimizer
        self.expert_registry = expert_registry or {}
        self.overlap_comm = overlap_comm
        self.cpu_offload = cpu_offload
        self.nvme_offload = nvme_offload
        self.partition_size = partition_size
        
        # Distributed setup
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Partition tracking
        self.param_partitions: Dict[str, List[torch.Tensor]] = {}
        self.gradient_partitions: Dict[str, List[torch.Tensor]] = {}
        self.optimizer_state_partitions: Dict[str, Any] = {}
        
        # Expert-specific tracking for MoE/MoD
        self.expert_param_map: Dict[str, List[str]] = defaultdict(list)
        self.expert_active_state: Dict[str, bool] = {}
        
        # Communication buffers
        self.gradient_buffer: Optional[torch.Tensor] = None
        self.param_buffer: Optional[torch.Tensor] = None
        
        # Hooks
        self._gradient_hooks: List[Any] = []
        self._forward_hooks: List[Any] = []
        self._backward_hooks: List[Any] = []
        
        self._initialize_stage()
    
    def _initialize_stage(self):
        """Initialize the selected ZeRO stage"""
        if self.stage == ZeROStage.DISABLED:
            return
        
        # Build expert parameter map for LuminaAI
        self._build_expert_param_map()
        
        if self.stage == ZeROStage.OPTIMIZER_STATES:
            self._init_zero_stage_1()
        elif self.stage == ZeROStage.GRADIENTS:
            self._init_zero_stage_2()
        elif self.stage == ZeROStage.PARAMETERS:
            self._init_zero_stage_3()
    
    def _build_expert_param_map(self):
        """Map parameters to experts for MoE/MoD-aware partitioning"""
        for expert_name, expert_module in self.expert_registry.items():
            for param_name, param in expert_module.named_parameters():
                full_name = f"{expert_name}.{param_name}"
                self.expert_param_map[expert_name].append(full_name)
                self.expert_active_state[expert_name] = True
    
    def _init_zero_stage_1(self):
        """
        ZeRO Stage 1: Partition optimizer states across ranks
        """
        print(f"[Rank {self.rank}] Initializing ZeRO Stage 1: Optimizer State Partitioning")
        
        # Partition optimizer states
        self._partition_optimizer_states()
        
        # Register post-backward hooks for gradient accumulation
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_hook(self._create_gradient_hook(param))
    
    def _init_zero_stage_2(self):
        """
        ZeRO Stage 2: Partition optimizer states + gradients across ranks
        """
        print(f"[Rank {self.rank}] Initializing ZeRO Stage 2: Optimizer + Gradient Partitioning")
        
        # Initialize Stage 1 first
        self._partition_optimizer_states()
        
        # Partition gradients
        self._partition_gradients()
        
        # Register hooks for gradient reduction
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_hook(self._create_gradient_partition_hook(param))
    
    def _init_zero_stage_3(self):
        """
        ZeRO Stage 3: Partition optimizer states + gradients + parameters across ranks
        """
        print(f"[Rank {self.rank}] Initializing ZeRO Stage 3: Full Parameter Partitioning")
        
        # Initialize Stage 2 first
        self._partition_optimizer_states()
        self._partition_gradients()
        
        # Partition parameters (most memory-intensive)
        self._partition_parameters()
        
        # Register forward/backward hooks for parameter gathering
        self._register_parameter_hooks()
    
    def _partition_optimizer_states(self):
        """Partition optimizer states across ranks"""
        optimizer_states = self.optimizer.state_dict()['state']
        
        params_per_rank = len(optimizer_states) // self.world_size
        start_idx = self.rank * params_per_rank
        end_idx = start_idx + params_per_rank if self.rank < self.world_size - 1 else len(optimizer_states)
        
        # Store local partition
        for param_id in range(start_idx, end_idx):
            if param_id in optimizer_states:
                self.optimizer_state_partitions[param_id] = optimizer_states[param_id]
        
        print(f"[Rank {self.rank}] Partitioned optimizer states: {len(self.optimizer_state_partitions)} local states")
    
    def _partition_gradients(self):
        """Partition gradients across ranks with expert awareness"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Check if parameter belongs to an expert
                expert_name = self._get_expert_for_param(name)
                
                # Assign rank based on hash (consistent partitioning)
                param_rank = hash(name) % self.world_size
                
                if param_rank == self.rank:
                    self.gradient_partitions[name] = []
                    
                    # Expert-specific logging
                    if expert_name:
                        print(f"[Rank {self.rank}] Assigned gradient partition: {name} (Expert: {expert_name})")
    
    def _partition_parameters(self):
        """Partition parameters across ranks with expert-aware allocation"""
        for name, param in self.model.named_parameters():
            param_rank = hash(name) % self.world_size
            
            if param_rank == self.rank:
                # Keep local copy
                self.param_partitions[name] = [param.data.clone()]
            else:
                # Free memory, will gather on demand
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
    
    def _get_expert_for_param(self, param_name: str) -> Optional[str]:
        """Find which expert a parameter belongs to"""
        for expert_name, param_list in self.expert_param_map.items():
            if any(param_name in p for p in param_list):
                return expert_name
        return None
    
    def _create_gradient_hook(self, param: torch.Tensor):
        """Create gradient accumulation hook for ZeRO-1"""
        def hook(grad):
            if self.overlap_comm and dist.is_initialized():
                # Asynchronously reduce gradients
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
            return grad
        return hook
    
    def _create_gradient_partition_hook(self, param: torch.Tensor):
        """Create gradient partitioning hook for ZeRO-2"""
        def hook(grad):
            if dist.is_initialized():
                # Reduce scatter: each rank gets its partition
                output = torch.empty_like(grad)
                dist.reduce_scatter_tensor(output, grad)
                return output
            return grad
        return hook
    
    def _register_parameter_hooks(self):
        """Register hooks for parameter gathering in ZeRO-3"""
        def forward_pre_hook(module, input):
            """Gather parameters before forward pass"""
            for name, param in module.named_parameters(recurse=False):
                if name in self.param_partitions:
                    # Gather full parameter from all ranks
                    self._gather_parameter(name, param)
        
        def forward_post_hook(module, input, output):
            """Release parameters after forward pass"""
            for name, param in module.named_parameters(recurse=False):
                if name not in self.param_partitions:
                    # Free non-local parameters
                    param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            return output
        
        # Register hooks on all modules
        for module in self.model.modules():
            self._forward_hooks.append(module.register_forward_pre_hook(forward_pre_hook))
            self._forward_hooks.append(module.register_forward_hook(forward_post_hook))
    
    def _gather_parameter(self, name: str, param: torch.Tensor):
        """Gather parameter from all ranks"""
        if not dist.is_initialized():
            return
        
        # Create buffer for gathering
        full_param = torch.empty(param.shape, dtype=param.dtype, device=param.device)
        
        # All-gather parameter
        dist.all_gather_into_tensor(full_param, param.data)
        param.data = full_param
    
    def step(self):
        """Execute optimizer step with ZeRO partitioning"""
        if self.stage == ZeROStage.DISABLED:
            self.optimizer.step()
            return
        
        # Synchronize gradients if needed
        if self.stage == ZeROStage.OPTIMIZER_STATES:
            self._sync_gradients()
        
        # Update local partition
        self.optimizer.step()
        
        # Broadcast updated parameters if ZeRO-3
        if self.stage == ZeROStage.PARAMETERS:
            self._broadcast_parameters()
    
    def _sync_gradients(self):
        """Synchronize gradients across ranks for ZeRO-1"""
        if dist.is_initialized():
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= self.world_size
    
    def _broadcast_parameters(self):
        """Broadcast updated parameters from owning rank"""
        for name, param in self.model.named_parameters():
            param_rank = hash(name) % self.world_size
            if dist.is_initialized():
                dist.broadcast(param.data, src=param_rank)
    
    def set_expert_active(self, expert_name: str, active: bool):
        """
        Hook for LuminaAI to enable/disable experts dynamically.
        Useful for mixture-of-depth routing.
        """
        if expert_name in self.expert_active_state:
            self.expert_active_state[expert_name] = active
            print(f"[Rank {self.rank}] Expert '{expert_name}' active state: {active}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Return memory statistics for current ZeRO stage"""
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9
        
        return {
            'stage': self.stage.value,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'num_param_partitions': len(self.param_partitions),
            'num_gradient_partitions': len(self.gradient_partitions),
            'num_optimizer_partitions': len(self.optimizer_state_partitions),
        }
    
    def cleanup(self):
        """Remove all hooks and clean up resources"""
        for hook in self._gradient_hooks + self._forward_hooks + self._backward_hooks:
            hook.remove()
        
        self._gradient_hooks.clear()
        self._forward_hooks.clear()
        self._backward_hooks.clear()


class ZeROConfig:
    """Configuration for ZeRO optimization"""
    
    def __init__(
        self,
        stage: int = 2,
        overlap_comm: bool = True,
        cpu_offload: bool = False,
        nvme_offload: bool = False,
        partition_size: int = 1e9,
        contiguous_gradients: bool = True,
        reduce_bucket_size: int = 5e8,
    ):
        self.stage = stage
        self.overlap_comm = overlap_comm
        self.cpu_offload = cpu_offload
        self.nvme_offload = nvme_offload
        self.partition_size = partition_size
        self.contiguous_gradients = contiguous_gradients
        self.reduce_bucket_size = reduce_bucket_size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'stage': self.stage,
            'overlap_comm': self.overlap_comm,
            'cpu_offload': self.cpu_offload,
            'nvme_offload': self.nvme_offload,
            'partition_size': self.partition_size,
            'contiguous_gradients': self.contiguous_gradients,
            'reduce_bucket_size': self.reduce_bucket_size,
        }


def create_zero_manager(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ZeROConfig,
    expert_registry: Optional[Dict[str, torch.nn.Module]] = None,
) -> ZeROStageManager:
    """
    Factory function to create ZeRO stage manager.
    
    Example usage:
        config = ZeROConfig(stage=2, cpu_offload=True)
        zero_manager = create_zero_manager(model, optimizer, config, expert_registry)
    """
    return ZeROStageManager(
        stage=config.stage,
        model=model,
        optimizer=optimizer,
        expert_registry=expert_registry,
        overlap_comm=config.overlap_comm,
        cpu_offload=config.cpu_offload,
        nvme_offload=config.nvme_offload,
        partition_size=config.partition_size,
    )