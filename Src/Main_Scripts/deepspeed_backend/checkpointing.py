"""
Checkpointing utilities for ZeRO-compatible training.
Handles partial and full checkpoint save/load with expert registry support.
"""

import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Optional, Any, List
import json
import time
import shutil


class ZeROCheckpoint:
    """
    Checkpoint manager for ZeRO-optimized models.
    Handles distributed checkpointing with expert awareness.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        zero_stage: int = 0,
        expert_registry: Optional[Dict] = None,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_rng_state: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.zero_stage = zero_stage
        self.expert_registry = expert_registry or {}
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_rng_state = save_rng_state
        
        # Distributed info
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        print(f"[Rank {self.rank}] Checkpoint manager initialized at {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metadata: Optional[Dict] = None,
        async_save: bool = False,
    ) -> str:
        """
        Save full checkpoint with ZeRO stage awareness.
        
        Args:
            model: PyTorch model
            optimizer: ZeRO-compatible optimizer
            scheduler: Learning rate scheduler
            step: Current training step
            epoch: Current epoch
            metadata: Additional metadata to save
            async_save: Save asynchronously (experimental)
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_step_{step}_epoch_{epoch}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        metadata = metadata or {}
        metadata.update({
            'step': step,
            'epoch': epoch,
            'zero_stage': self.zero_stage,
            'world_size': self.world_size,
            'timestamp': time.time(),
        })
        
        # Save model state
        if self.zero_stage == 0:
            # No partitioning, save full model
            if self.rank == 0:
                torch.save(model.state_dict(), checkpoint_path / "model.pt")
        else:
            # ZeRO partitioning: each rank saves its partition
            rank_model_path = checkpoint_path / f"model_rank_{self.rank}.pt"
            torch.save(model.state_dict(), rank_model_path)
        
        # Save optimizer state
        if self.save_optimizer and optimizer is not None:
            if self.zero_stage >= 1:
                # Save partitioned optimizer state
                rank_optim_path = checkpoint_path / f"optimizer_rank_{self.rank}.pt"
                torch.save(optimizer.state_dict(), rank_optim_path)
            else:
                if self.rank == 0:
                    torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        
        # Save scheduler state
        if self.save_scheduler and scheduler is not None and self.rank == 0:
            torch.save(scheduler.state_dict(), checkpoint_path / "scheduler.pt")
        
        # Save expert registry
        if self.expert_registry and self.rank == 0:
            expert_metadata = {
                name: {
                    'num_parameters': sum(p.numel() for p in module.parameters()),
                    'dtype': str(next(module.parameters()).dtype),
                }
                for name, module in self.expert_registry.items()
            }
            with open(checkpoint_path / "expert_registry.json", 'w') as f:
                json.dump(expert_metadata, f, indent=2)
        
        # Save RNG state
        if self.save_rng_state:
            rng_state = {
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            torch.save(rng_state, checkpoint_path / f"rng_state_rank_{self.rank}.pt")
        
        # Save metadata
        if self.rank == 0:
            with open(checkpoint_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()
        
        print(f"[Rank {self.rank}] Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint with ZeRO stage awareness.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model: PyTorch model to load into
            optimizer: ZeRO-compatible optimizer
            scheduler: Learning rate scheduler
            strict: Strict loading (fail on missing keys)
        
        Returns:
            Metadata dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load metadata
        metadata = {}
        if (checkpoint_path / "metadata.json").exists():
            with open(checkpoint_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
        
        # Validate ZeRO stage
        if metadata.get('zero_stage') != self.zero_stage:
            print(f"[Warning] Checkpoint ZeRO stage ({metadata.get('zero_stage')}) "
                  f"!= current ZeRO stage ({self.zero_stage})")
        
        # Load model state
        if self.zero_stage == 0:
            # Load full model
            model_path = checkpoint_path / "model.pt"
            if model_path.exists():
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=strict)
        else:
            # Load rank-specific partition
            rank_model_path = checkpoint_path / f"model_rank_{self.rank}.pt"
            if rank_model_path.exists():
                state_dict = torch.load(rank_model_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=strict)
            else:
                print(f"[Warning] No model checkpoint found for rank {self.rank}")
        
        # Load optimizer state
        if optimizer is not None:
            if self.zero_stage >= 1:
                rank_optim_path = checkpoint_path / f"optimizer_rank_{self.rank}.pt"
                if rank_optim_path.exists():
                    optim_state = torch.load(rank_optim_path, map_location='cpu')
                    optimizer.load_state_dict(optim_state)
            else:
                optim_path = checkpoint_path / "optimizer.pt"
                if optim_path.exists():
                    optim_state = torch.load(optim_path, map_location='cpu')
                    optimizer.load_state_dict(optim_state)
        
        # Load scheduler state
        if scheduler is not None and self.rank == 0:
            scheduler_path = checkpoint_path / "scheduler.pt"
            if scheduler_path.exists():
                scheduler_state = torch.load(scheduler_path, map_location='cpu')
                scheduler.load_state_dict(scheduler_state)
        
        # Load RNG state
        if self.save_rng_state:
            rng_path = checkpoint_path / f"rng_state_rank_{self.rank}.pt"
            if rng_path.exists():
                rng_state = torch.load(rng_path, map_location='cpu')
                torch.set_rng_state(rng_state['torch_rng_state'])
                if rng_state['cuda_rng_state'] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng_state['cuda_rng_state'])
        
        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()
        
        print(f"[Rank {self.rank}] Loaded checkpoint: {checkpoint_path}")
        return metadata
    
    def save_expert_checkpoint(
        self,
        expert_name: str,
        expert_module: torch.nn.Module,
        step: int = 0,
    ) -> str:
        """
        Save individual expert checkpoint (useful for MoE/MoD).
        
        Args:
            expert_name: Name of expert
            expert_module: Expert module
            step: Current training step
        
        Returns:
            Path to saved expert checkpoint
        """
        expert_dir = self.checkpoint_dir / "experts" / expert_name
        expert_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = expert_dir / f"expert_{expert_name}_step_{step}.pt"
        
        expert_state = {
            'state_dict': expert_module.state_dict(),
            'step': step,
            'num_parameters': sum(p.numel() for p in expert_module.parameters()),
        }
        
        torch.save(expert_state, checkpoint_path)
        print(f"[Rank {self.rank}] Saved expert '{expert_name}': {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_expert_checkpoint(
        self,
        expert_name: str,
        expert_module: torch.nn.Module,
        step: Optional[int] = None,
        strict: bool = True,
    ) -> bool:
        """
        Load individual expert checkpoint.
        
        Args:
            expert_name: Name of expert
            expert_module: Expert module to load into
            step: Specific step to load (None = latest)
            strict: Strict loading
        
        Returns:
            True if successful, False otherwise
        """
        expert_dir = self.checkpoint_dir / "experts" / expert_name
        
        if not expert_dir.exists():
            print(f"[Warning] No checkpoints found for expert '{expert_name}'")
            return False
        
        # Find checkpoint
        if step is not None:
            checkpoint_path = expert_dir / f"expert_{expert_name}_step_{step}.pt"
        else:
            # Find latest checkpoint
            checkpoints = list(expert_dir.glob(f"expert_{expert_name}_step_*.pt"))
            if not checkpoints:
                return False
            checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        
        if not checkpoint_path.exists():
            return False
        
        # Load state
        expert_state = torch.load(checkpoint_path, map_location='cpu')
        expert_module.load_state_dict(expert_state['state_dict'], strict=strict)
        
        print(f"[Rank {self.rank}] Loaded expert '{expert_name}' from {checkpoint_path}")
        return True
    
    def consolidate_zero_checkpoint(
        self,
        checkpoint_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Consolidate ZeRO-partitioned checkpoint into single file.
        Should be run on rank 0 after all ranks have saved.
        
        Args:
            checkpoint_path: Path to partitioned checkpoint
            output_path: Output path for consolidated checkpoint
        
        Returns:
            Path to consolidated checkpoint
        """
        if self.rank != 0:
            print("[Warning] Checkpoint consolidation should only run on rank 0")
            return ""
        
        checkpoint_path = Path(checkpoint_path)
        output_path = Path(output_path) if output_path else checkpoint_path / "consolidated"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Consolidate model state
        print("Consolidating model state...")
        model_states = []
        for rank in range(self.world_size):
            rank_path = checkpoint_path / f"model_rank_{rank}.pt"
            if rank_path.exists():
                model_states.append(torch.load(rank_path, map_location='cpu'))
        
        if model_states:
            # Merge all rank states
            consolidated_model = {}
            for state in model_states:
                consolidated_model.update(state)
            
            torch.save(consolidated_model, output_path / "model.pt")
            print(f"Saved consolidated model: {output_path / 'model.pt'}")
        
        # Consolidate optimizer state
        if self.save_optimizer:
            print("Consolidating optimizer state...")
            optim_states = []
            for rank in range(self.world_size):
                rank_path = checkpoint_path / f"optimizer_rank_{rank}.pt"
                if rank_path.exists():
                    optim_states.append(torch.load(rank_path, map_location='cpu'))
            
            if optim_states:
                # Merge optimizer states
                consolidated_optim = {
                    'state': {},
                    'param_groups': optim_states[0].get('param_groups', []),
                }
                
                for state in optim_states:
                    if 'state' in state:
                        consolidated_optim['state'].update(state['state'])
                
                torch.save(consolidated_optim, output_path / "optimizer.pt")
                print(f"Saved consolidated optimizer: {output_path / 'optimizer.pt'}")
        
        # Copy metadata and other files
        for file in ['metadata.json', 'scheduler.pt', 'expert_registry.json']:
            src = checkpoint_path / file
            if src.exists():
                shutil.copy(src, output_path / file)
        
        print(f"Checkpoint consolidation complete: {output_path}")
        return str(output_path)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata"""
        checkpoints = []
        
        for ckpt_dir in sorted(self.checkpoint_dir.glob("checkpoint_*")):
            metadata_path = ckpt_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    metadata['path'] = str(ckpt_dir)
                    checkpoints.append(metadata)
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """
        Remove old checkpoints, keeping only the last N.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        if self.rank != 0:
            return
        
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            return
        
        # Sort by step
        checkpoints.sort(key=lambda x: x.get('step', 0))
        
        # Remove old checkpoints
        for ckpt in checkpoints[:-keep_last_n]:
            ckpt_path = Path(ckpt['path'])
            if ckpt_path.exists():
                shutil.rmtree(ckpt_path)
                print(f"Removed old checkpoint: {ckpt_path}")
        
        print(f"Kept last {keep_last_n} checkpoints")


class IncrementalCheckpoint:
    """
    Incremental checkpointing for long-running training.
    Only saves changed parameters to reduce I/O overhead.
    """
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.previous_state: Optional[Dict] = None
        self.checkpoint_count = 0
    
    def save_incremental(
        self,
        model: torch.nn.Module,
        step: int,
    ) -> str:
        """
        Save only parameters that have changed since last checkpoint.
        
        Args:
            model: PyTorch model
            step: Current training step
        
        Returns:
            Path to incremental checkpoint
        """
        current_state = model.state_dict()
        
        if self.previous_state is None:
            # First checkpoint: save everything
            changes = current_state
        else:
            # Find changed parameters
            changes = {}
            for key, value in current_state.items():
                if key not in self.previous_state:
                    changes[key] = value
                elif not torch.equal(value, self.previous_state[key]):
                    changes[key] = value
        
        # Save incremental changes
        checkpoint_path = self.checkpoint_dir / f"incremental_step_{step}.pt"
        torch.save({
            'changes': changes,
            'step': step,
            'is_full': self.previous_state is None,
        }, checkpoint_path)
        
        self.previous_state = current_state
        self.checkpoint_count += 1
        
        print(f"Saved incremental checkpoint: {checkpoint_path} "
              f"({len(changes)} changed parameters)")
        
        return str(checkpoint_path)
    
    def restore_from_incremental(
        self,
        model: torch.nn.Module,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Restore model from sequence of incremental checkpoints.
        
        Args:
            model: PyTorch model to restore
            checkpoint_dir: Directory containing incremental checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.checkpoint_dir
        
        # Find all incremental checkpoints
        checkpoints = sorted(
            checkpoint_dir.glob("incremental_step_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        if not checkpoints:
            print("[Warning] No incremental checkpoints found")
            return
        
        # Apply checkpoints in sequence
        state_dict = {}
        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict.update(ckpt['changes'])
        
        model.load_state_dict(state_dict)
        print(f"Restored model from {len(checkpoints)} incremental checkpoints")