import torch
import time
from training.trainer import EnhancedConversationTrainer

def profile_training_loop_overhead(trainer, train_dataloader, num_batches=10):
    """Profile where time is actually spent in the training loop."""
    import time
    
    print("\n" + "="*80)
    print("PROFILING TRAINING LOOP OVERHEAD")
    print("="*80)
    
    timings = {
        'data_loading': [],
        'to_device': [],
        'forward': [],
        'backward': [],
        'optimizer': [],
        'logging': [],
        'metrics': [],
        'total_step': []
    }
    
    trainer.model.train()
    
    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx >= num_batches:
            break
        
        step_start = time.perf_counter()
        
        # Data loading (already done by dataloader, just measure transfer)
        to_device_start = time.perf_counter()
        batch = {k: v.to(trainer.device, non_blocking=True) for k, v in batch.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings['to_device'].append((time.perf_counter() - to_device_start) * 1000)
        
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels', input_ids)
        
        # Forward pass
        forward_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        output = trainer.model(input_ids, attention_mask)
        
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        loss_dict = trainer.compute_loss(logits, labels, None)
        loss = loss_dict['loss']
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings['forward'].append((time.perf_counter() - forward_start) * 1000)
        
        # Backward pass
        backward_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings['backward'].append((time.perf_counter() - backward_start) * 1000)
        
        # Optimizer step
        optimizer_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings['optimizer'].append((time.perf_counter() - optimizer_start) * 1000)
        
        # Metrics collection (simulate)
        metrics_start = time.perf_counter()
        _ = loss.item()
        _ = loss_dict['accuracy'].item()
        timings['metrics'].append((time.perf_counter() - metrics_start) * 1000)
        
        # Total step time
        timings['total_step'].append((time.perf_counter() - step_start) * 1000)
    
    # Print results
    print(f"\nResults (averaged over {num_batches} batches):")
    print("="*80)
    print(f"{'Operation':<20} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'% of Total':<12}")
    print("-"*80)
    
    total_avg = sum(timings['total_step']) / len(timings['total_step'])
    
    for name, times in timings.items():
        if not times:
            continue
        avg = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        pct = (avg / total_avg * 100) if total_avg > 0 else 0
        
        print(f"{name:<20} {avg:>10.2f}  {min_time:>10.2f}  {max_time:>10.2f}  {pct:>10.1f}%")
    
    print("="*80)
    
    # Compute time breakdown
    compute_time = sum(timings['forward']) / len(timings['forward']) + sum(timings['backward']) / len(timings['backward'])
    overhead_time = total_avg - compute_time
    
    print(f"\nTiming Breakdown:")
    print(f"  Pure compute (forward+backward): {compute_time:.2f}ms ({compute_time/total_avg*100:.1f}%)")
    print(f"  Overhead (optimizer+logging+metrics): {overhead_time:.2f}ms ({overhead_time/total_avg*100:.1f}%)")
    
    # Calculate what tokens/sec SHOULD be
    if timings['forward']:
        num_tokens = input_ids.numel()
        compute_throughput = num_tokens / (compute_time / 1000)
        wall_clock_throughput = num_tokens / (total_avg / 1000)
        
        print(f"\nThroughput Analysis:")
        print(f"  Based on compute time: {compute_throughput:.0f} tokens/sec")
        print(f"  Based on wall clock: {wall_clock_throughput:.0f} tokens/sec")
        print(f"  Overhead penalty: {(1 - wall_clock_throughput/compute_throughput)*100:.1f}%")
    
    print("="*80)
    
    # Recommendations
    print("\nRecommendations:")
    
    overhead_pct = (overhead_time / total_avg * 100)
    if overhead_pct > 30:
        print(f"  ⚠️  HIGH OVERHEAD ({overhead_pct:.1f}%) - Reduce logging frequency")
    
    to_device_avg = sum(timings['to_device']) / len(timings['to_device'])
    if to_device_avg > 5:
        print(f"  ⚠️  SLOW DATA TRANSFER ({to_device_avg:.1f}ms) - Use pin_memory=True in DataLoader")
    
    optimizer_avg = sum(timings['optimizer']) / len(timings['optimizer'])
    if optimizer_avg > compute_time * 0.2:
        print(f"  ⚠️  SLOW OPTIMIZER ({optimizer_avg:.1f}ms) - Consider fused optimizer")
    
    forward_avg = sum(timings['forward']) / len(timings['forward'])
    backward_avg = sum(timings['backward']) / len(timings['backward'])
    if backward_avg > forward_avg * 2:
        print(f"  ℹ️  Backward is {backward_avg/forward_avg:.1f}x slower than forward (normal for transformers)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    from model import DeepSeekConfig, DeepSeekTransformer
    from tokenizers import Tokenizer
    
    # Create dummy trainer
    config = DeepSeekConfig(
        hidden_size=128,
        num_experts=8,
        moe_top_k=2,
        use_moe=True,
        use_cuda_moe=True
    )
    
    model = DeepSeekTransformer(config)
    tokenizer = Tokenizer.from_file("your_tokenizer.json")  # Replace with your tokenizer
    
    trainer = EnhancedConversationTrainer(model, tokenizer, config, logger=None)
    
    # Create dummy dataloader
    from torch.utils.data import DataLoader, TensorDataset
    dummy_data = torch.randint(0, config.vocab_size, (100, 256))
    dataset = TensorDataset(dummy_data)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Profile it
    profile_training_loop_overhead(trainer, dataloader, num_batches=10)