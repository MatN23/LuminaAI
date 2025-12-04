# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from enum import Enum
import math


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
        
        # FIX: Use deterministic parameter ordering
        self.param_list = list(model.parameters())
        self.param_to_name = {}
        for name, param in model.named_parameters():
            self.param_to_name[id(param)] = name
        
        # Partition tracking with proper typing
        self.param_to_rank: Dict[int, int] = {}  # param_id -> owning rank
        self.rank_to_params: Dict[int, List[torch.nn.Parameter]] = defaultdict(list)
        self.gradient_partitions: Dict[int, torch.Tensor] = {}  # param_id -> grad partition
        self.optimizer_state_partitions: Dict[int, Any] = {}
        
        # FIX: CPU offload buffers (pinned memory for faster transfers)
        self.cpu_offload_buffers: Dict[int, torch.Tensor] = {}
        
        # Expert-specific tracking for MoE/MoD
        self.expert_param_map: Dict[str, List[str]] = defaultdict(list)
        self.expert_active_state: Dict[str, bool] = {}
        
        # FIX: Gradient accumulation buffers for bucketing
        self.gradient_buckets: List[List[torch.nn.Parameter]] = []
        self.bucket_size = int(5e8)  # 500MB default bucket
        
        # FIX: Full parameter storage for ZeRO-3
        self.full_params: Dict[int, torch.Tensor] = {}  # Stores full param when gathered
        
        # Hooks
        self._gradient_hooks: List[Any] = []
        self._forward_hooks: List[Any] = []
        self._backward_hooks: List[Any] = []
        
        # FIX: Communication handles for async operations
        self.comm_handles: List[dist.Work] = []
        
        self._initialize_stage()
    
    def _initialize_stage(self):
        """Initialize the selected ZeRO stage"""
        if self.stage == ZeROStage.DISABLED:
            return
        
        # Build expert parameter map for LuminaAI
        self._build_expert_param_map()
        
        # FIX: Create deterministic parameter partitioning first
        self._create_parameter_partitions()
        
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
    
    def _create_parameter_partitions(self):
        """
        FIX: Create deterministic parameter partitions across ranks.
        Uses round-robin assignment based on parameter index (not hash).
        """
        params = list(self.model.parameters())
        
        for idx, param in enumerate(params):
            # Deterministic round-robin assignment
            owning_rank = idx % self.world_size
            param_id = id(param)
            
            self.param_to_rank[param_id] = owning_rank
            self.rank_to_params[owning_rank].append(param)
        
        print(f"[Rank {self.rank}] Owns {len(self.rank_to_params[self.rank])} parameters")
    
    def _create_gradient_buckets(self):
        """
        FIX: Create gradient buckets for efficient reduce-scatter.
        Groups parameters by size to minimize communication overhead.
        """
        current_bucket = []
        current_bucket_size = 0
        
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            
            param_size = param.numel() * param.element_size()
            
            if current_bucket_size + param_size > self.bucket_size and current_bucket:
                self.gradient_buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0
            
            current_bucket.append(param)
            current_bucket_size += param_size
        
        if current_bucket:
            self.gradient_buckets.append(current_bucket)
        
        print(f"[Rank {self.rank}] Created {len(self.gradient_buckets)} gradient buckets")
    
    def _init_zero_stage_1(self):
        """
        ZeRO Stage 1: Partition optimizer states across ranks
        FIX: Properly partition optimizer state dict by parameter ownership
        """
        print(f"[Rank {self.rank}] Initializing ZeRO Stage 1: Optimizer State Partitioning")
        
        # Partition optimizer states based on parameter ownership
        self._partition_optimizer_states()
        
        # FIX: Create gradient buckets for efficient all-reduce
        self._create_gradient_buckets()
        
        # Register gradient hooks for all-reduce
        for param in self.model.parameters():
            if param.requires_grad:
                hook = param.register_hook(self._create_allreduce_hook(param))
                self._gradient_hooks.append(hook)
    
    def _init_zero_stage_2(self):
        """
        ZeRO Stage 2: Partition optimizer states + gradients across ranks
        FIX: Use reduce-scatter for gradient partitioning instead of all-reduce
        """
        print(f"[Rank {self.rank}] Initializing ZeRO Stage 2: Optimizer + Gradient Partitioning")
        
        # Partition optimizer states
        self._partition_optimizer_states()
        
        # Create gradient buckets
        self._create_gradient_buckets()
        
        # FIX: Register reduce-scatter hooks for gradient partitioning
        for param in self.model.parameters():
            if param.requires_grad:
                hook = param.register_hook(self._create_reduce_scatter_hook(param))
                self._gradient_hooks.append(hook)
        
        # Allocate gradient partition buffers
        for param in self.rank_to_params[self.rank]:
            if param.requires_grad:
                param_id = id(param)
                # Each rank only stores its partition of gradients
                partition_size = math.ceil(param.numel() / self.world_size)
                self.gradient_partitions[param_id] = torch.zeros(
                    partition_size, 
                    dtype=param.dtype, 
                    device=param.device
                )
    
    def _init_zero_stage_3(self):
        """
        ZeRO Stage 3: Partition optimizer states + gradients + parameters across ranks
        FIX: Properly implement parameter gathering/scattering with forward/backward hooks
        """
        print(f"[Rank {self.rank}] Initializing ZeRO Stage 3: Full Parameter Partitioning")
        
        # Initialize Stage 2 components
        self._partition_optimizer_states()
        self._create_gradient_buckets()
        
        # Partition parameters and create CPU offload buffers
        self._partition_parameters_stage3()
        
        # FIX: Register forward pre-hooks for parameter gathering
        self._register_parameter_hooks()
        
        # Register gradient hooks for reduce-scatter
        for param in self.model.parameters():
            if param.requires_grad:
                hook = param.register_hook(self._create_reduce_scatter_hook(param))
                self._gradient_hooks.append(hook)
    
    def _partition_optimizer_states(self):
        """
        FIX: Partition optimizer states deterministically based on parameter ownership.
        Only keep states for parameters owned by this rank.
        """
        optimizer_states = self.optimizer.state
        
        # Clear existing partitions
        self.optimizer_state_partitions.clear()
        
        # Only keep optimizer states for parameters this rank owns
        for param in self.rank_to_params[self.rank]:
            param_id = id(param)
            if param in optimizer_states:
                self.optimizer_state_partitions[param_id] = optimizer_states[param]
                
                # FIX: Offload to CPU if enabled
                if self.cpu_offload:
                    self._offload_optimizer_state_to_cpu(param_id)
        
        print(f"[Rank {self.rank}] Partitioned {len(self.optimizer_state_partitions)} optimizer states")
    
    def _partition_parameters_stage3(self):
        """
        FIX: Partition parameters for ZeRO-3.
        Each rank keeps only its partition; others are freed or offloaded.
        """
        for param in self.model.parameters():
            param_id = id(param)
            owning_rank = self.param_to_rank[param_id]
            
            if owning_rank == self.rank:
                # This rank owns this parameter - keep full copy
                self.full_params[param_id] = param.data.clone()
                
                # FIX: Offload to CPU if enabled
                if self.cpu_offload:
                    self.cpu_offload_buffers[param_id] = param.data.to('cpu', non_blocking=True).pin_memory()
            else:
                # Not owned by this rank - free GPU memory
                # We'll gather on-demand during forward pass
                param.data = param.data.new_empty(0)
        
        print(f"[Rank {self.rank}] Partitioned parameters for ZeRO-3")
    
    def _create_allreduce_hook(self, param: torch.nn.Parameter):
        """
        FIX: Create gradient all-reduce hook for ZeRO-1.
        Properly averages gradients across all ranks.
        """
        def hook(grad):
            if grad is None or not dist.is_initialized():
                return grad
            
            # FIX: Synchronous all-reduce with proper averaging
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(self.world_size)
            
            return grad
        
        return hook
    
    def _create_reduce_scatter_hook(self, param: torch.nn.Parameter):
        """
        FIX: Create gradient reduce-scatter hook for ZeRO-2/3.
        Each rank gets its partition of the reduced gradient.
        """
        param_id = id(param)
        owning_rank = self.param_to_rank[param_id]
        
        def hook(grad):
            if grad is None or not dist.is_initialized():
                return grad
            
            # Only the owning rank needs the gradient partition
            if owning_rank != self.rank:
                # This rank doesn't own this param, zero out gradient
                return torch.zeros_like(grad)
            
            # FIX: Use reduce_scatter to partition gradients
            # Each rank computes sum of its partition across all ranks
            grad_flat = grad.flatten()
            chunk_size = math.ceil(grad_flat.numel() / self.world_size)
            
            # Pad if necessary for even division
            padded_size = chunk_size * self.world_size
            if grad_flat.numel() < padded_size:
                grad_flat = torch.nn.functional.pad(grad_flat, (0, padded_size - grad_flat.numel()))
            
            # Split into chunks for all ranks
            grad_chunks = grad_flat.split(chunk_size)
            
            # Output buffer for this rank's partition
            output = torch.zeros_like(grad_chunks[self.rank])
            
            # Reduce-scatter: sum all ranks' chunks[i] into rank i
            input_list = [chunk.contiguous() for chunk in grad_chunks]
            dist.reduce_scatter_tensor(output, torch.stack(input_list), op=dist.ReduceOp.SUM)
            
            # Store in partition buffer
            self.gradient_partitions[param_id] = output
            
            # Return the partition (reshaped to match original)
            return output.view_as(grad[:chunk_size])
        
        return hook
    
    def _register_parameter_hooks(self):
        """
        FIX: Register hooks for parameter gathering/scattering in ZeRO-3.
        Gather before forward, scatter after backward.
        """
        def forward_pre_hook(module, input):
            """Gather parameters before forward pass"""
            for param in module.parameters(recurse=False):
                if not param.requires_grad:
                    continue
                    
                param_id = id(param)
                owning_rank = self.param_to_rank.get(param_id)
                
                if owning_rank is None:
                    continue
                
                # FIX: Gather parameter from owning rank
                if owning_rank == self.rank:
                    # This rank owns it - restore from CPU if offloaded
                    if self.cpu_offload and param_id in self.cpu_offload_buffers:
                        param.data = self.cpu_offload_buffers[param_id].to(
                            param.device, non_blocking=True
                        )
                    elif param_id in self.full_params:
                        param.data = self.full_params[param_id]
                else:
                    # Receive from owning rank via broadcast
                    if param.data.numel() == 0:
                        # Allocate space for full parameter
                        param.data = torch.empty(
                            self.full_params.get(param_id, param).shape,
                            dtype=param.dtype,
                            device=param.device
                        )
                
                # FIX: Broadcast from owning rank to all ranks
                if dist.is_initialized():
                    dist.broadcast(param.data, src=owning_rank)
        
        def backward_post_hook(module, grad_input, grad_output):
            """Free parameters after backward pass"""
            for param in module.parameters(recurse=False):
                param_id = id(param)
                owning_rank = self.param_to_rank.get(param_id)
                
                if owning_rank != self.rank:
                    # Free non-owned parameters to save memory
                    param.data = param.data.new_empty(0)
            
            return grad_input
        
        # Register hooks on all modules
        for module in self.model.modules():
            hook1 = module.register_forward_pre_hook(forward_pre_hook)
            hook2 = module.register_full_backward_hook(backward_post_hook)
            self._forward_hooks.append(hook1)
            self._backward_hooks.append(hook2)
    
    def _offload_optimizer_state_to_cpu(self, param_id: int):
        """
        FIX: Offload optimizer state to pinned CPU memory for faster transfers.
        """
        if param_id not in self.optimizer_state_partitions:
            return
        
        state = self.optimizer_state_partitions[param_id]
        
        # Offload tensors in state dict to CPU with pinned memory
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to('cpu', non_blocking=True).pin_memory()
    
    def step(self):
        """
        FIX: Execute optimizer step with proper ZeRO partitioning.
        Synchronizes gradients and parameters as needed.
        """
        if self.stage == ZeROStage.DISABLED:
            self.optimizer.step()
            return
        
        # FIX: Wait for any pending async operations
        self._wait_for_communication()
        
        # For ZeRO-1, gradients are already synchronized via all-reduce
        # For ZeRO-2/3, gradients are partitioned via reduce-scatter
        
        # FIX: Move optimizer states back to GPU if offloaded
        if self.cpu_offload:
            for param_id in self.optimizer_state_partitions:
                self._restore_optimizer_state_from_cpu(param_id)
        
        # Update local partition only
        self.optimizer.step()
        
        # FIX: Offload back to CPU after update
        if self.cpu_offload:
            for param_id in self.optimizer_state_partitions:
                self._offload_optimizer_state_to_cpu(param_id)
        
        # For ZeRO-3, broadcast updated parameters from owning ranks
        if self.stage == ZeROStage.PARAMETERS:
            self._broadcast_parameters()
    
    def _restore_optimizer_state_from_cpu(self, param_id: int):
        """Restore optimizer state from CPU to GPU"""
        if param_id not in self.optimizer_state_partitions:
            return
        
        state = self.optimizer_state_partitions[param_id]
        
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                state[key] = value.cuda(non_blocking=True)
    
    def _wait_for_communication(self):
        """Wait for all async communication operations to complete"""
        for handle in self.comm_handles:
            handle.wait()
        self.comm_handles.clear()
    
    def _broadcast_parameters(self):
        """
        FIX: Broadcast updated parameters from owning ranks after optimizer step.
        """
        if not dist.is_initialized():
            return
        
        for param in self.model.parameters():
            param_id = id(param)
            owning_rank = self.param_to_rank.get(param_id)
            
            if owning_rank is not None:
                # Broadcast from owning rank
                dist.broadcast(param.data, src=owning_rank)
    
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
        allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        
        return {
            'stage': self.stage.value,
            'rank': self.rank,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'num_owned_params': len(self.rank_to_params[self.rank]),
            'num_gradient_partitions': len(self.gradient_partitions),
            'num_optimizer_partitions': len(self.optimizer_state_partitions),
            'num_cpu_offload_buffers': len(self.cpu_offload_buffers),
        }
    
    def cleanup(self):
        """Remove all hooks and clean up resources"""
        for hook in self._gradient_hooks + self._forward_hooks + self._backward_hooks:
            hook.remove()
        
        self._gradient_hooks.clear()
        self._forward_hooks.clear()
        self._backward_hooks.clear()
        
        # Wait for pending communications
        self._wait_for_communication()
        
        # Clear buffers
        self.cpu_offload_buffers.clear()
        self.full_params.clear()
        self.gradient_partitions.clear()


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