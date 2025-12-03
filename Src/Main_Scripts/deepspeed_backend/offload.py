"""
Offload Manager for LuminaAI DeepSpeed Backend
Handles CPU and NVMe offloading for optimizer states, parameters, and gradients.
"""

import torch
import os
import mmap
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import pickle
import threading


@dataclass
class OffloadConfig:
    """Configuration for offloading strategies"""
    device: str = 'cpu'  # 'cpu' or 'nvme'
    offload_optimizer_states: bool = True
    offload_parameters: bool = False
    offload_gradients: bool = False
    nvme_path: Optional[str] = None
    buffer_size: int = 1e9  # 1GB buffer for NVMe
    pin_memory: bool = True
    async_offload: bool = True
    

class CPUOffloadManager:
    """
    Manages offloading to CPU memory with pinned memory support.
    Reduces GPU memory usage for optimizer states and parameters.
    """
    
    def __init__(self, config: OffloadConfig, expert_registry: Optional[Dict] = None):
        self.config = config
        self.expert_registry = expert_registry or {}
        
        # Storage for offloaded tensors
        self.cpu_optimizer_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.cpu_parameters: Dict[str, torch.Tensor] = {}
        self.cpu_gradients: Dict[str, torch.Tensor] = {}
        
        # Pinned memory pool for faster transfers
        self.pin_memory = config.pin_memory
        self.pinned_buffers: Dict[str, torch.Tensor] = {}
        
        # Async transfer tracking
        self.pending_transfers: List[torch.cuda.Stream] = []
        
        print(f"[CPUOffload] Initialized with pin_memory={self.pin_memory}")
    
    def offload_optimizer_state(self, param_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Offload optimizer state for a parameter to CPU.
        
        Args:
            param_id: Parameter identifier
            state: Optimizer state dict (momentum, variance, etc.)
        
        Returns:
            Offloaded state dict with CPU tensors
        """
        cpu_state = {}
        
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if self.pin_memory:
                    # Allocate pinned memory for faster transfer
                    cpu_tensor = torch.empty(
                        value.shape, 
                        dtype=value.dtype, 
                        pin_memory=True
                    )
                    cpu_tensor.copy_(value, non_blocking=True)
                else:
                    cpu_tensor = value.cpu()
                
                cpu_state[key] = cpu_tensor
            else:
                cpu_state[key] = value
        
        self.cpu_optimizer_states[param_id] = cpu_state
        return cpu_state
    
    def restore_optimizer_state(self, param_id: int, device: torch.device) -> Dict[str, Any]:
        """
        Restore optimizer state from CPU to GPU.
        
        Args:
            param_id: Parameter identifier
            device: Target device (usually cuda)
        
        Returns:
            Restored state dict on target device
        """
        if param_id not in self.cpu_optimizer_states:
            return {}
        
        cpu_state = self.cpu_optimizer_states[param_id]
        gpu_state = {}
        
        for key, value in cpu_state.items():
            if isinstance(value, torch.Tensor):
                gpu_state[key] = value.to(device, non_blocking=True)
            else:
                gpu_state[key] = value
        
        return gpu_state
    
    def offload_parameter(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """
        Offload parameter to CPU and return placeholder.
        
        Args:
            name: Parameter name
            param: Parameter tensor
        
        Returns:
            Empty GPU tensor (placeholder)
        """
        if self.pin_memory:
            cpu_param = torch.empty(param.shape, dtype=param.dtype, pin_memory=True)
            cpu_param.copy_(param.data, non_blocking=True)
        else:
            cpu_param = param.data.cpu()
        
        self.cpu_parameters[name] = cpu_param
        
        # Return empty placeholder to free GPU memory
        return torch.empty(0, dtype=param.dtype, device=param.device)
    
    def restore_parameter(self, name: str, device: torch.device) -> Optional[torch.Tensor]:
        """
        Restore parameter from CPU to GPU.
        
        Args:
            name: Parameter name
            device: Target device
        
        Returns:
            Restored parameter tensor
        """
        if name not in self.cpu_parameters:
            return None
        
        cpu_param = self.cpu_parameters[name]
        gpu_param = cpu_param.to(device, non_blocking=True)
        
        return gpu_param
    
    def offload_gradient(self, name: str, grad: torch.Tensor):
        """Offload gradient to CPU"""
        if self.pin_memory:
            cpu_grad = torch.empty(grad.shape, dtype=grad.dtype, pin_memory=True)
            cpu_grad.copy_(grad, non_blocking=True)
        else:
            cpu_grad = grad.cpu()
        
        self.cpu_gradients[name] = cpu_grad
    
    def restore_gradient(self, name: str, device: torch.device) -> Optional[torch.Tensor]:
        """Restore gradient from CPU to GPU"""
        if name not in self.cpu_gradients:
            return None
        
        return self.cpu_gradients[name].to(device, non_blocking=True)
    
    def get_memory_savings(self) -> Dict[str, float]:
        """Calculate memory savings from offloading"""
        optimizer_memory = sum(
            sum(t.numel() * t.element_size() for t in state.values() if isinstance(t, torch.Tensor))
            for state in self.cpu_optimizer_states.values()
        ) / 1e9
        
        param_memory = sum(
            p.numel() * p.element_size() for p in self.cpu_parameters.values()
        ) / 1e9
        
        grad_memory = sum(
            g.numel() * g.element_size() for g in self.cpu_gradients.values()
        ) / 1e9
        
        return {
            'optimizer_states_gb': optimizer_memory,
            'parameters_gb': param_memory,
            'gradients_gb': grad_memory,
            'total_gb': optimizer_memory + param_memory + grad_memory,
        }
    
    def clear(self):
        """Clear all offloaded data"""
        self.cpu_optimizer_states.clear()
        self.cpu_parameters.clear()
        self.cpu_gradients.clear()
        self.pinned_buffers.clear()


class NVMeOffloadManager:
    """
    Manages offloading to NVMe storage for extreme memory savings.
    Useful for training massive models that don't fit in CPU memory.
    """
    
    def __init__(self, config: OffloadConfig, expert_registry: Optional[Dict] = None):
        self.config = config
        self.expert_registry = expert_registry or {}
        
        # Validate NVMe path
        if not config.nvme_path:
            raise ValueError("nvme_path must be specified for NVMe offloading")
        
        self.nvme_path = Path(config.nvme_path)
        self.nvme_path.mkdir(parents=True, exist_ok=True)
        
        # Memory-mapped files for efficient I/O
        self.mmap_files: Dict[str, mmap.mmap] = {}
        self.file_handles: Dict[str, Any] = {}
        
        # Buffer for batched writes
        self.buffer_size = config.buffer_size
        self.write_buffer: Dict[str, List[Tuple[str, torch.Tensor]]] = defaultdict(list)
        
        # Threading for async operations
        self.async_offload = config.async_offload
        self.write_lock = threading.Lock()
        
        print(f"[NVMeOffload] Initialized at {self.nvme_path}")
    
    def offload_optimizer_state(self, param_id: int, state: Dict[str, Any]) -> str:
        """
        Offload optimizer state to NVMe.
        
        Args:
            param_id: Parameter identifier
            state: Optimizer state dict
        
        Returns:
            Path to saved state file
        """
        file_path = self.nvme_path / f"optimizer_state_{param_id}.pt"
        
        # Convert tensors to CPU before saving
        cpu_state = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                cpu_state[key] = value.cpu()
            else:
                cpu_state[key] = value
        
        # Save to disk
        if self.async_offload:
            self._async_save(file_path, cpu_state)
        else:
            torch.save(cpu_state, file_path)
        
        return str(file_path)
    
    def restore_optimizer_state(self, param_id: int, device: torch.device) -> Dict[str, Any]:
        """
        Restore optimizer state from NVMe.
        
        Args:
            param_id: Parameter identifier
            device: Target device
        
        Returns:
            Restored optimizer state
        """
        file_path = self.nvme_path / f"optimizer_state_{param_id}.pt"
        
        if not file_path.exists():
            return {}
        
        cpu_state = torch.load(file_path, map_location='cpu')
        
        # Move to target device
        gpu_state = {}
        for key, value in cpu_state.items():
            if isinstance(value, torch.Tensor):
                gpu_state[key] = value.to(device)
            else:
                gpu_state[key] = value
        
        return gpu_state
    
    def offload_parameter(self, name: str, param: torch.Tensor) -> str:
        """
        Offload parameter to NVMe.
        
        Args:
            name: Parameter name
            param: Parameter tensor
        
        Returns:
            Path to saved parameter file
        """
        # Sanitize filename
        safe_name = name.replace('/', '_').replace('.', '_')
        file_path = self.nvme_path / f"param_{safe_name}.pt"
        
        # Save CPU tensor
        cpu_param = param.data.cpu()
        
        if self.async_offload:
            self._async_save(file_path, cpu_param)
        else:
            torch.save(cpu_param, file_path)
        
        return str(file_path)
    
    def restore_parameter(self, name: str, device: torch.device) -> Optional[torch.Tensor]:
        """
        Restore parameter from NVMe.
        
        Args:
            name: Parameter name
            device: Target device
        
        Returns:
            Restored parameter tensor
        """
        safe_name = name.replace('/', '_').replace('.', '_')
        file_path = self.nvme_path / f"param_{safe_name}.pt"
        
        if not file_path.exists():
            return None
        
        cpu_param = torch.load(file_path, map_location='cpu')
        return cpu_param.to(device)
    
    def _async_save(self, path: Path, data: Any):
        """Asynchronously save data to disk"""
        def _save_worker():
            with self.write_lock:
                torch.save(data, path)
        
        if self.async_offload:
            thread = threading.Thread(target=_save_worker)
            thread.start()
        else:
            _save_worker()
    
    def offload_expert_states(self, expert_name: str, expert_module: torch.nn.Module):
        """
        Offload entire expert module to NVMe (useful for MoE with many experts).
        
        Args:
            expert_name: Name of expert
            expert_module: Expert module to offload
        """
        expert_dir = self.nvme_path / f"expert_{expert_name}"
        expert_dir.mkdir(exist_ok=True)
        
        # Save all parameters
        state_dict = expert_module.state_dict()
        for param_name, param in state_dict.items():
            param_path = expert_dir / f"{param_name}.pt"
            torch.save(param.cpu(), param_path)
        
        print(f"[NVMeOffload] Offloaded expert '{expert_name}' to {expert_dir}")
    
    def restore_expert_states(self, expert_name: str, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Restore expert module from NVMe.
        
        Args:
            expert_name: Name of expert
            device: Target device
        
        Returns:
            State dict with restored parameters
        """
        expert_dir = self.nvme_path / f"expert_{expert_name}"
        
        if not expert_dir.exists():
            return {}
        
        state_dict = {}
        for param_file in expert_dir.glob("*.pt"):
            param_name = param_file.stem
            param = torch.load(param_file, map_location='cpu')
            state_dict[param_name] = param.to(device)
        
        return state_dict
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Calculate disk usage for offloaded data"""
        total_bytes = sum(
            f.stat().st_size for f in self.nvme_path.rglob("*.pt")
        )
        
        return {
            'total_gb': total_bytes / 1e9,
            'num_files': len(list(self.nvme_path.rglob("*.pt"))),
            'path': str(self.nvme_path),
        }
    
    def cleanup(self):
        """Close all file handles and clean up"""
        for mmap_file in self.mmap_files.values():
            mmap_file.close()
        
        for handle in self.file_handles.values():
            handle.close()
        
        self.mmap_files.clear()
        self.file_handles.clear()
    
    def clear_disk(self):
        """Delete all offloaded files (use with caution!)"""
        import shutil
        if self.nvme_path.exists():
            shutil.rmtree(self.nvme_path)
            self.nvme_path.mkdir(parents=True, exist_ok=True)


class HybridOffloadManager:
    """
    Combines CPU and NVMe offloading with intelligent tiering.
    Hot data stays in CPU memory, cold data goes to NVMe.
    """
    
    def __init__(
        self, 
        config: OffloadConfig, 
        expert_registry: Optional[Dict] = None,
        cpu_threshold: float = 0.8,  # Use CPU until 80% full
    ):
        self.config = config
        self.expert_registry = expert_registry or {}
        self.cpu_threshold = cpu_threshold
        
        # Initialize both managers
        self.cpu_manager = CPUOffloadManager(config, expert_registry)
        
        nvme_config = OffloadConfig(
            device='nvme',
            nvme_path=config.nvme_path,
            buffer_size=config.buffer_size,
            async_offload=config.async_offload,
        )
        self.nvme_manager = NVMeOffloadManager(nvme_config, expert_registry)
        
        # Access tracking for intelligent tiering
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, int] = {}
        self.timestep = 0
    
    def offload_parameter(self, name: str, param: torch.Tensor) -> str:
        """
        Intelligently offload to CPU or NVMe based on memory pressure.
        
        Returns:
            'cpu' or 'nvme' indicating where data was stored
        """
        self.timestep += 1
        self.access_counts[name] += 1
        self.last_access[name] = self.timestep
        
        # Check CPU memory usage
        import psutil
        cpu_memory_percent = psutil.virtual_memory().percent / 100.0
        
        if cpu_memory_percent < self.cpu_threshold:
            # Use CPU offload
            self.cpu_manager.offload_parameter(name, param)
            return 'cpu'
        else:
            # Spill to NVMe
            self.nvme_manager.offload_parameter(name, param)
            return 'nvme'
    
    def restore_parameter(self, name: str, device: torch.device) -> Optional[torch.Tensor]:
        """Restore parameter from CPU or NVMe"""
        self.timestep += 1
        self.access_counts[name] += 1
        self.last_access[name] = self.timestep
        
        # Try CPU first (faster)
        param = self.cpu_manager.restore_parameter(name, device)
        if param is not None:
            return param
        
        # Fall back to NVMe
        return self.nvme_manager.restore_parameter(name, device)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for both storage tiers"""
        return {
            'cpu': self.cpu_manager.get_memory_savings(),
            'nvme': self.nvme_manager.get_disk_usage(),
            'access_counts': dict(self.access_counts),
            'timestep': self.timestep,
        }


def create_offload_manager(
    config: OffloadConfig,
    expert_registry: Optional[Dict[str, torch.nn.Module]] = None,
) -> Any:
    """
    Factory function to create appropriate offload manager.
    
    Args:
        config: Offload configuration
        expert_registry: LuminaAI expert registry for MoE/MoD awareness
    
    Returns:
        CPU, NVMe, or Hybrid offload manager
    """
    if config.device == 'cpu':
        return CPUOffloadManager(config, expert_registry)
    elif config.device == 'nvme':
        return NVMeOffloadManager(config, expert_registry)
    elif config.device == 'hybrid':
        return HybridOffloadManager(config, expert_registry)
    else:
        raise ValueError(f"Unknown offload device: {config.device}")