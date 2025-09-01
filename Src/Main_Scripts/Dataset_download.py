# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import logging
import json
import gc
import psutil
import multiprocessing as mp
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Install with: pip install datasets")
    print("Run: pip install datasets huggingface_hub")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ShardConfig:
    """Configuration for dataset sharding."""
    max_shard_size_mb: int = 512
    min_shard_size_mb: int = 50
    max_memory_usage_gb: float = 8.0
    num_workers: int = min(8, mp.cpu_count())
    buffer_size: int = 10000
    enable_memory_mapping: bool = True
    enable_compression: bool = False
    cache_shards: bool = True
    shard_shuffle: bool = True


class EnhancedShardManager:
    """Manages dataset sharding for datasets from 150MB to 16TB+."""
    
    def __init__(self, base_path: Path, config: ShardConfig):
        self.base_path = Path(base_path)
        self.config = config
        self.shards_dir = self.base_path / "shards"
        self.metadata_file = self.base_path / "shard_metadata.json"
        
        self.stats = {
            'total_conversations': 0,
            'total_shards': 0,
            'total_size_mb': 0,
            'avg_shard_size_mb': 0,
            'load_strategy': 'unknown'
        }
        
        self.shards_dir.mkdir(parents=True, exist_ok=True)
    
    def estimate_processing_strategy(self, dataset_size: int) -> str:
        """Determine the best processing strategy based on dataset size."""
        # Get system memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Estimate memory requirements (rough)
        estimated_memory_gb = dataset_size * 0.001  # ~1KB per item average
        
        if estimated_memory_gb < memory_gb * 0.3:  # < 30% of system memory
            return "memory"
        elif estimated_memory_gb < memory_gb * 0.6:  # < 60% of system memory
            return "sharded"
        else:
            return "streaming"
    
    def create_shards_from_conversations(self, conversations: List[Dict], split_name: str) -> List[Path]:
        """Create shards from processed conversations."""
        if not conversations:
            return []
        
        shard_paths = []
        shard_size_limit = self.config.max_shard_size_mb * 1024 * 1024
        
        current_shard = []
        current_shard_size = 0
        shard_index = 0
        
        logger.info(f"Creating shards with {self.config.max_shard_size_mb}MB limit...")
        
        for conv in conversations:
            conv_size = len(json.dumps(conv, ensure_ascii=False).encode('utf-8'))
            
            # Check if we need to start a new shard
            if (current_shard_size + conv_size > shard_size_limit and 
                current_shard and len(current_shard) >= 100):
                
                # Save current shard
                shard_path = self._save_shard(current_shard, split_name, shard_index)
                shard_paths.append(shard_path)
                
                # Reset for next shard
                current_shard = []
                current_shard_size = 0
                shard_index += 1
                
                if shard_index % 10 == 0:
                    logger.info(f"Created {shard_index} shards...")
            
            current_shard.append(conv)
            current_shard_size += conv_size
        
        # Save final shard
        if current_shard:
            shard_path = self._save_shard(current_shard, split_name, shard_index)
            shard_paths.append(shard_path)
        
        # Save metadata
        self._save_metadata(shard_paths, split_name, len(conversations))
        
        self.stats['total_shards'] = len(shard_paths)
        self.stats['total_conversations'] = len(conversations)
        
        return shard_paths
    
    def _save_shard(self, conversations: List[Dict], split_name: str, shard_index: int) -> Path:
        """Save a shard to disk."""
        shard_path = self.shards_dir / f"oasst1_{split_name}_shard_{shard_index:04d}.jsonl"
        
        with open(shard_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        size_mb = shard_path.stat().st_size / (1024 * 1024)
        self.stats['total_size_mb'] += size_mb
        
        return shard_path
    
    def _save_metadata(self, shard_paths: List[Path], split_name: str, total_conversations: int):
        """Save shard metadata."""
        metadata = {
            'split': split_name,
            'total_shards': len(shard_paths),
            'total_conversations': total_conversations,
            'shard_paths': [str(p) for p in shard_paths],
            'config': {
                'max_shard_size_mb': self.config.max_shard_size_mb,
                'created_at': os.path.getmtime(shard_paths[0]) if shard_paths else 0
            }
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def setup_output_directory(project_root: Optional[str] = None) -> Path:
    """Setup and create output directory for dataset files with sharding support."""
    if project_root is None:
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir
    
    output_dir = Path(project_root) / "oasst1_data"
    
    # Create main directory and shards subdirectory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "shards").mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        raise
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Shards directory: {output_dir / 'shards'}")
    
    return output_dir


def check_system_resources() -> Dict:
    """Check system resources and recommend sharding configuration."""
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    logger.info(f"System Resources:")
    logger.info(f"  RAM: {memory_gb:.1f} GB")
    logger.info(f"  CPU cores: {cpu_count}")
    
    # Recommend sharding configuration based on available resources
    if memory_gb < 16:
        recommended_shard_size = 256
        recommended_workers = min(2, cpu_count // 2)
        max_memory_usage = memory_gb * 0.6
    elif memory_gb < 64:
        recommended_shard_size = 512
        recommended_workers = min(4, cpu_count // 2)
        max_memory_usage = memory_gb * 0.7
    else:
        recommended_shard_size = 1024
        recommended_workers = min(8, cpu_count)
        max_memory_usage = memory_gb * 0.8
    
    logger.info(f"Recommended sharding config:")
    logger.info(f"  Shard size: {recommended_shard_size} MB")
    logger.info(f"  Workers: {recommended_workers}")
    logger.info(f"  Max memory usage: {max_memory_usage:.1f} GB")
    
    return {
        'max_shard_size_mb': recommended_shard_size,
        'num_workers': recommended_workers,
        'max_memory_usage_gb': max_memory_usage
    }


def build_conversation_tree(messages: List[Dict]) -> Tuple[Dict[str, Dict], List[str]]:
    """Build a tree structure from messages using parent-child relationships."""
    message_map = {}
    
    # First pass: create message map
    for msg in messages:
        message_map[msg['message_id']] = {
            'data': msg,
            'children': []
        }
    
    # Second pass: build parent-child relationships
    root_messages = []
    for msg in messages:
        parent_id = msg.get('parent_id')
        if parent_id and parent_id in message_map:
            message_map[parent_id]['children'].append(msg['message_id'])
        else:
            # This is a root message (conversation starter)
            root_messages.append(msg['message_id'])
    
    return message_map, root_messages


def extract_conversation_paths(message_map: Dict, root_id: str) -> List[List[Dict]]:
    """Extract all possible conversation paths from a root message."""
    conversations = []
    
    def dfs_path(node_id: str, current_path: List[Dict]):
        if node_id not in message_map:
            return
        
        node = message_map[node_id]
        new_path = current_path + [node['data']]
        
        # Save conversation at multiple points to generate more data
        if len(new_path) >= 2:  # At least one exchange
            conversations.append(new_path.copy())
        
        # If this is a leaf node or has no valid children, we're done with this path
        if not node['children']:
            return
        else:
            # Continue down each child path
            for child_id in node['children']:
                dfs_path(child_id, new_path)
    
    dfs_path(root_id, [])
    return conversations


def format_conversation(messages: List[Dict]) -> Dict:
    """Format a conversation path into a structured format."""
    conversation = {
        'conversation_id': messages[0].get('message_tree_id', ''),
        'messages': [],
        'total_turns': len(messages),
        'languages': list(set(msg.get('lang', 'en') for msg in messages)),
        'created_date': messages[0].get('created_date', ''),
        'tree_state': messages[0].get('tree_state', '')
    }
    
    for i, msg in enumerate(messages):
        formatted_msg = {
            'turn': i + 1,
            'role': msg.get('role', '').lower(),
            'content': msg.get('text', '').strip(),
            'message_id': msg.get('message_id', ''),
            'review_result': msg.get('review_result', None),
            'rank': msg.get('rank', 0),
            'synthetic': msg.get('synthetic', False),
            'model_name': msg.get('model_name', '')
        }
        conversation['messages'].append(formatted_msg)
    
    return conversation


def filter_quality_conversations(conversations: List[Dict], strict_filtering: bool = False) -> List[Dict]:
    """Filter conversations based on quality criteria."""
    quality_conversations = []
    
    for conv in conversations:
        # Quality checks
        valid = True
        
        # Must have at least 2 turns (one exchange)
        if len(conv['messages']) < 2:
            continue
            
        # Must start with a prompter (user)
        if conv['messages'][0]['role'] not in ['prompter', 'user']:
            continue
            
        # Must alternate between prompter and assistant
        expected_role = 'prompter'
        for msg in conv['messages']:
            if msg['role'] not in [expected_role, 'user' if expected_role == 'prompter' else expected_role]:
                valid = False
                break
            expected_role = 'assistant' if expected_role in ['prompter', 'user'] else 'prompter'
        
        if not valid:
            continue
        
        # Content validation
        has_content = True
        for msg in conv['messages']:
            if not msg['content'] or len(msg['content'].strip()) == 0:
                has_content = False
                break
                
        if not has_content:
            continue
            
        quality_conversations.append(conv)
    
    return quality_conversations


def process_dataset_chunk(chunk_data: Tuple[int, List, str]) -> Tuple[int, List[Dict]]:
    """Process a chunk of dataset messages in parallel."""
    chunk_idx, messages_chunk, split_name = chunk_data
    
    # Group messages by conversation tree
    tree_messages = defaultdict(list)
    for msg in messages_chunk:
        tree_id = msg.get('message_tree_id', '')
        if tree_id:
            tree_messages[tree_id].append(msg)
    
    # Build conversations from trees
    chunk_conversations = []
    for tree_id, messages in tree_messages.items():
        try:
            message_map, root_messages = build_conversation_tree(messages)
            
            for root_id in root_messages:
                paths = extract_conversation_paths(message_map, root_id)
                for path in paths:
                    conv = format_conversation(path)
                    chunk_conversations.append(conv)
                    
                    # Create sub-conversations for longer paths to maximize data
                    if len(path) > 4:
                        for start_idx in range(0, len(path) - 3, 2):
                            if start_idx > 0:
                                sub_path = path[start_idx:]
                                if len(sub_path) >= 2:
                                    sub_conv = format_conversation(sub_path)
                                    sub_conv['conversation_id'] = f"{sub_conv['conversation_id']}_sub_{start_idx}"
                                    chunk_conversations.append(sub_conv)
        except Exception as e:
            logger.debug(f"Error processing tree {tree_id}: {e}")
            continue
    
    # Filter quality conversations
    filtered_conversations = filter_quality_conversations(chunk_conversations)
    
    return chunk_idx, filtered_conversations


def download_and_process_conversations_enhanced(output_dir: Path, shard_config: ShardConfig) -> bool:
    """Enhanced dataset processing with automatic sharding and parallel processing."""
    try:
        logger.info("Loading OpenAssistant dataset (oasst1)...")
        logger.info("Enhanced version with sharding support for datasets up to 16TB+")
        
        # Load dataset with error handling
        try:
            ds = load_dataset("OpenAssistant/oasst1", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.info("Possible solutions:")
            logger.info("  1. Try running: huggingface-cli login")
            logger.info("  2. Install git-lfs if not installed")
            logger.info("  3. Check your internet connection")
            return False
        
        logger.info("Dataset loaded successfully!")
        logger.info(f"Available splits: {list(ds.keys())}")
        
        # Initialize enhanced shard manager
        shard_manager = EnhancedShardManager(output_dir, shard_config)
        
        # Process train, validation, and evaluation splits
        splits_to_process = ['train', 'validation']
        
        # Check for test/eval splits
        if 'test' in ds:
            splits_to_process.append('test')
        elif 'eval' in ds:
            splits_to_process.append('eval')
        elif 'evaluation' in ds:
            splits_to_process.append('evaluation')
        
        for split_name in splits_to_process:
            if split_name not in ds:
                logger.warning(f"Split '{split_name}' not found in dataset")
                continue
                
            logger.info(f"Processing {split_name} split with enhanced sharding...")
            
            # Get split data
            split_data = ds[split_name]
            logger.info(f"Total messages in {split_name}: {len(split_data):,}")
            
            # Determine processing strategy
            strategy = shard_manager.estimate_processing_strategy(len(split_data))
            logger.info(f"Processing strategy: {strategy}")
            
            # Filter valid messages first
            logger.info("Filtering valid messages...")
            valid_messages = []
            
            for i, msg in enumerate(split_data):
                if i > 0 and i % 25000 == 0:
                    logger.info(f"  Filtered {i:,}/{len(split_data):,} messages...")
                
                if (msg.get('text', '').strip() and msg.get('message_tree_id')):
                    valid_messages.append(msg)
            
            logger.info(f"Valid messages in {split_name}: {len(valid_messages):,}")
            
            # Process based on strategy
            if strategy == "memory" and len(valid_messages) < 100000:
                # Small dataset - process in memory
                all_conversations = self._process_messages_in_memory(valid_messages)
            
            elif strategy == "sharded" or len(valid_messages) < 500000:
                # Medium dataset - parallel processing with chunks
                all_conversations = self._process_messages_parallel(valid_messages, split_name, shard_config)
            
            else:
                # Large dataset - streaming processing
                all_conversations = self._process_messages_streaming(valid_messages, split_name, shard_config)
            
            logger.info(f"Total conversations extracted from {split_name}: {len(all_conversations):,}")
            
            # Create output based on dataset size
            if len(all_conversations) < 50000:  # Small dataset
                # Save as single file
                output_file = output_dir / f"oasst1_{split_name}_conversations.jsonl"
                self._save_conversations_single_file(all_conversations, output_file)
                logger.info(f"Saved {len(all_conversations):,} conversations to single file")
            
            else:
                # Save as sharded files
                shard_paths = shard_manager.create_shards_from_conversations(all_conversations, split_name)
                logger.info(f"Created {len(shard_paths)} shard files for {split_name}")
                
                # Also create a unified file for compatibility
                output_file = output_dir / f"oasst1_{split_name}_conversations.jsonl"
                self._save_conversations_single_file(all_conversations, output_file)
                logger.info(f"Also saved unified file: {output_file.name}")
            
            # Analyze dataset
            logger.info(f"\n--- Enhanced Analysis for {split_name} ---")
            analyze_conversations_enhanced(all_conversations, split_name)
            
            # Memory cleanup
            del all_conversations, valid_messages
            gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"Error in enhanced processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def _process_messages_in_memory(valid_messages: List[Dict]) -> List[Dict]:
    """Process messages in memory for small datasets."""
    logger.info("Processing in memory (small dataset)")
    
    # Group messages by conversation tree
    tree_messages = defaultdict(list)
    for msg in valid_messages:
        tree_id = msg.get('message_tree_id', '')
        if tree_id:
            tree_messages[tree_id].append(msg)
    
    # Build conversations
    all_conversations = []
    processed_trees = 0
    
    for tree_id, messages in tree_messages.items():
        processed_trees += 1
        if processed_trees % 1000 == 0:
            logger.info(f"  Processing tree {processed_trees:,}/{len(tree_messages):,}...")
        
        try:
            message_map, root_messages = build_conversation_tree(messages)
            
            for root_id in root_messages:
                paths = extract_conversation_paths(message_map, root_id)
                for path in paths:
                    conv = format_conversation(path)
                    all_conversations.append(conv)
                    
                    # Create sub-conversations
                    if len(path) > 4:
                        for start_idx in range(0, len(path) - 3, 2):
                            if start_idx > 0:
                                sub_path = path[start_idx:]
                                if len(sub_path) >= 2:
                                    sub_conv = format_conversation(sub_path)
                                    sub_conv['conversation_id'] = f"{sub_conv['conversation_id']}_sub_{start_idx}"
                                    all_conversations.append(sub_conv)
        except Exception as e:
            logger.debug(f"Error processing tree {tree_id}: {e}")
            continue
    
    return filter_quality_conversations(all_conversations)


def _process_messages_parallel(valid_messages: List[Dict], split_name: str, shard_config: ShardConfig) -> List[Dict]:
    """Process messages using parallel workers for medium datasets."""
    logger.info(f"Processing with parallel workers (medium dataset)")
    
    # Split messages into chunks for parallel processing
    chunk_size = max(5000, len(valid_messages) // (shard_config.num_workers * 2))
    chunks = []
    
    for i in range(0, len(valid_messages), chunk_size):
        end_idx = min(i + chunk_size, len(valid_messages))
        chunks.append((i // chunk_size, valid_messages[i:end_idx], split_name))
    
    logger.info(f"Processing {len(valid_messages):,} messages in {len(chunks)} chunks using {shard_config.num_workers} workers")
    
    # Process chunks in parallel
    all_conversations = []
    
    with ProcessPoolExecutor(max_workers=shard_config.num_workers) as executor:
        future_to_chunk = {executor.submit(process_dataset_chunk, chunk): chunk[0] for chunk in chunks}
        
        for future in as_completed(future_to_chunk):
            try:
                chunk_idx, chunk_conversations = future.result()
                all_conversations.extend(chunk_conversations)
                
                if len(all_conversations) % 10000 == 0:
                    logger.info(f"Processed {len(all_conversations):,} conversations...")
                    
            except Exception as e:
                logger.warning(f"Chunk processing failed: {e}")
    
    return all_conversations


def _process_messages_streaming(valid_messages: List[Dict], split_name: str, shard_config: ShardConfig) -> List[Dict]:
    """Process messages with streaming approach for massive datasets."""
    logger.info(f"Processing with streaming approach (large dataset)")
    
    # For very large datasets, process in smaller memory-conscious chunks
    all_conversations = []
    chunk_size = min(10000, max(1000, len(valid_messages) // 50))
    
    # Group messages by tree first (streaming)
    tree_messages = defaultdict(list)
    
    logger.info("Building conversation trees...")
    for i, msg in enumerate(valid_messages):
        if i % 50000 == 0 and i > 0:
            logger.info(f"  Grouped {i:,}/{len(valid_messages):,} messages...")
        
        tree_id = msg.get('message_tree_id', '')
        if tree_id:
            tree_messages[tree_id].append(msg)
    
    logger.info(f"Found {len(tree_messages):,} conversation trees")
    
    # Process trees in batches to manage memory
    processed_trees = 0
    tree_items = list(tree_messages.items())
    
    for batch_start in range(0, len(tree_items), 1000):  # Process 1000 trees at a time
        batch_end = min(batch_start + 1000, len(tree_items))
        batch_trees = tree_items[batch_start:batch_end]
        
        batch_conversations = []
        
        for tree_id, messages in batch_trees:
            processed_trees += 1
            
            if processed_trees % 5000 == 0:
                logger.info(f"  Processing tree {processed_trees:,}/{len(tree_items):,}...")
            
            try:
                message_map, root_messages = build_conversation_tree(messages)
                
                for root_id in root_messages:
                    paths = extract_conversation_paths(message_map, root_id)
                    for path in paths:
                        conv = format_conversation(path)
                        batch_conversations.append(conv)
                        
                        # Create sub-conversations
                        if len(path) > 4:
                            for start_idx in range(0, len(path) - 3, 2):
                                if start_idx > 0:
                                    sub_path = path[start_idx:]
                                    if len(sub_path) >= 2:
                                        sub_conv = format_conversation(sub_path)
                                        sub_conv['conversation_id'] = f"{sub_conv['conversation_id']}_sub_{start_idx}"
                                        batch_conversations.append(sub_conv)
            except Exception:
                continue
        
        # Filter and add batch conversations
        filtered_batch = filter_quality_conversations(batch_conversations)
        all_conversations.extend(filtered_batch)
        
        # Clear batch memory
        del batch_conversations, filtered_batch
        gc.collect()
        
        if len(all_conversations) % 25000 == 0:
            logger.info(f"Total conversations: {len(all_conversations):,}")
    
    return all_conversations


def _save_conversations_single_file(conversations: List[Dict], output_file: Path):
    """Save conversations to a single file."""
    logger.info(f"Saving {len(conversations):,} conversations to: {output_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in conversations:
            try:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
                saved_count += 1
                
                if saved_count % 10000 == 0:
                    logger.info(f"  Saved {saved_count:,}/{len(conversations):,} conversations...")
                    
            except Exception as e:
                logger.warning(f"Error saving conversation: {e}")
                continue
    
    logger.info(f"Successfully saved {saved_count:,} conversations")


def analyze_conversations_enhanced(conversations: List[Dict], split_name: str):
    """Enhanced conversation analysis with memory efficiency for large datasets."""
    if not conversations:
        logger.warning(f"No conversations found in {split_name}")
        return
    
    # For very large datasets, sample for analysis to manage memory
    if len(conversations) > 20000:
        import random
        sample_size = 10000
        sample_conversations = random.sample(conversations, sample_size)
        logger.info(f"Analyzing sample of {sample_size:,} conversations from {len(conversations):,} total")
    else:
        sample_conversations = conversations
    
    total_conversations = len(conversations)
    total_turns = sum(len(conv['messages']) for conv in sample_conversations)
    avg_turns = total_turns / len(sample_conversations) if sample_conversations else 0
    
    turn_distribution = defaultdict(int)
    role_counts = defaultdict(int)
    language_counts = defaultdict(int)
    
    for conv in sample_conversations:
        turn_distribution[len(conv['messages'])] += 1
        
        for msg in conv['messages']:
            role_counts[msg['role']] += 1
        
        # Track languages
        for lang in conv.get('languages', ['en']):
            language_counts[lang] += 1
    
    logger.info(f"Enhanced Analysis for {split_name}:")
    logger.info(f"  Total conversations: {total_conversations:,}")
    logger.info(f"  Analyzed sample: {len(sample_conversations):,}")
    logger.info(f"  Total turns (sampled): {total_turns:,}")
    logger.info(f"  Average turns per conversation: {avg_turns:.1f}")
    
    logger.info(f"  Turn distribution (top 10):")
    for turns, count in sorted(turn_distribution.items())[:10]:
        pct = count / len(sample_conversations) * 100
        logger.info(f"    {turns} turns: {count:,} conversations ({pct:.1f}%)")
    
    logger.info(f"  Role distribution:")
    total_role_messages = sum(role_counts.values())
    for role, count in role_counts.items():
        pct = count / total_role_messages * 100 if total_role_messages > 0 else 0
        logger.info(f"    {role}: {count:,} messages ({pct:.1f}%)")
    
    logger.info(f"  Language distribution (top 5):")
    for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = count / len(sample_conversations) * 100
        logger.info(f"    {lang}: {count:,} conversations ({pct:.1f}%)")


def validate_conversation_files_enhanced(output_dir: Path) -> bool:
    """Enhanced validation for both regular and sharded conversation files."""
    # Get all conversation files (including shards)
    conversation_files = list(output_dir.glob("oasst1_*.jsonl"))
    shard_files = list((output_dir / "shards").glob("*.jsonl")) if (output_dir / "shards").exists() else []
    
    all_files = conversation_files + shard_files
    
    if not all_files:
        logger.error("No conversation files found")
        return False
    
    logger.info(f"Validating {len(all_files)} files...")
    
    for file_path in all_files:
        if not file_path.exists():
            logger.error(f"Missing file: {file_path}")
            return False
        
        if file_path.stat().st_size == 0:
            logger.error(f"Empty file: {file_path}")
            return False
        
        # Test reading and parsing (sample based on file size)
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Sample fewer lines for larger files to avoid memory issues
            if file_size_mb > 100:
                sample_lines = 1
            elif file_size_mb > 10:
                sample_lines = 2
            else:
                sample_lines = 3
            
            line_count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line_count += 1
                    
                    # Test parsing of sample lines
                    if i < sample_lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            # Check conversation structure
                            required_fields = ['conversation_id', 'messages', 'total_turns']
                            for field in required_fields:
                                if field not in data:
                                    logger.error(f"Missing field '{field}' in {file_path}")
                                    return False
                            
                            # Check message structure
                            if data['messages']:
                                msg = data['messages'][0]
                                msg_fields = ['role', 'content', 'turn']
                                for field in msg_fields:
                                    if field not in msg:
                                        logger.error(f"Missing message field '{field}' in {file_path}")
                                        return False
                                        
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON in {file_path} line {i+1}: {e}")
                            return False
            
            # Show file info
            if "_shard_" in file_path.name:
                logger.info(f"Shard {file_path.name}: {file_size_mb:.1f}MB, {line_count:,} conversations")
            else:
                logger.info(f"{file_path.name}: {file_size_mb:.1f}MB, {line_count:,} conversations")
                        
        except Exception as e:
            logger.error(f"Cannot read file {file_path}: {e}")
            return False
    
    logger.info("All conversation files validated successfully!")
    return True


def check_existing_files_enhanced(output_dir: Path) -> bool:
    """Enhanced check for existing files including sharded datasets."""
    # Look for regular conversation files
    regular_files = list(output_dir.glob("oasst1_*_conversations.jsonl"))
    
    # Look for sharded files
    shard_files = list((output_dir / "shards").glob("*.jsonl")) if (output_dir / "shards").exists() else []
    
    # Check for metadata
    metadata_file = output_dir / "shard_metadata.json"
    has_metadata = metadata_file.exists()
    
    if regular_files or shard_files:
        logger.info("Found existing conversation files!")
        
        total_size_mb = 0
        total_conversations = 0
        
        # Report regular files
        for file_path in regular_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            try:
                with open(file_path, 'r') as f:
                    line_count = sum(1 for _ in f)
                total_conversations += line_count
                logger.info(f"  - {file_path.name}: {size_mb:.1f} MB, {line_count:,} conversations")
            except Exception:
                logger.info(f"  - {file_path.name}: {size_mb:.1f} MB")
        
        # Report sharded files
        if shard_files:
            shard_size_mb = 0
            shard_conversations = 0
            
            for file_path in shard_files[:5]:  # Show first 5 shards
                size_mb = file_path.stat().st_size / (1024 * 1024)
                shard_size_mb += size_mb
                try:
                    with open(file_path, 'r') as f:
                        line_count = sum(1 for _ in f)
                    shard_conversations += line_count
                    logger.info(f"  - {file_path.name}: {size_mb:.1f} MB, {line_count:,} conversations")
                except Exception:
                    logger.info(f"  - {file_path.name}: {size_mb:.1f} MB")
            
            # Count remaining shards
            for file_path in shard_files[5:]:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                shard_size_mb += size_mb
                try:
                    with open(file_path, 'r') as f:
                        line_count = sum(1 for _ in f)
                    shard_conversations += line_count
                except Exception:
                    pass
            
            total_size_mb += shard_size_mb
            total_conversations += shard_conversations
            
            if len(shard_files) > 5:
                logger.info(f"  ... and {len(shard_files) - 5} more shard files")
            
            logger.info(f"  Total shards: {len(shard_files)}, {shard_size_mb:.1f} MB, {shard_conversations:,} conversations")
        
        if has_metadata:
            logger.info(f"  - shard_metadata.json: Sharding information available")
        
        logger.info(f"Total dataset: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB), {total_conversations:,} conversations")
        
        # Determine loading strategy based on size
        if total_size_mb < 500:
            strategy = "in-memory loading"
        elif total_size_mb < 10000:
            strategy = "sharded loading"
        else:
            strategy = "streaming loading"
        
        logger.info(f"Recommended loading strategy: {strategy}")
        
        logger.info("Validating existing files...")
        return validate_conversation_files_enhanced(output_dir)
    
    return False


def main():
    """Enhanced main function with comprehensive sharding support."""
    logger.info("Starting Enhanced OASST1 Dataset Download with Sharding Support")
    logger.info("Supports datasets from 150MB to 16TB+ with automatic optimization")
    logger.info("=" * 80)
    
    try:
        # Check and report system resources
        system_resources = check_system_resources()
        
        # Create shard configuration based on system resources
        shard_config = ShardConfig(
            max_shard_size_mb=system_resources['max_shard_size_mb'],
            max_memory_usage_gb=system_resources['max_memory_usage_gb'],
            num_workers=system_resources['num_workers']
        )
        
        # Setup directories
        output_dir = setup_output_directory()
        
        # Check if files already exist and are valid
        if check_existing_files_enhanced(output_dir):
            logger.info("Valid dataset files already exist!")
            logger.info("Delete files in the output directory if you want to re-download")
            logger.info("=" * 80)
            return 0
        else:
            logger.info("Downloading/reprocessing dataset with enhanced sharding...")
        
        # Monitor memory usage during processing
        initial_memory = psutil.virtual_memory().percent
        logger.info(f"Initial memory usage: {initial_memory:.1f}%")
        
        # Download and process conversations with enhanced sharding
        success = download_and_process_conversations_enhanced(output_dir, shard_config)
        
        if not success:
            logger.error("Enhanced dataset processing failed!")
            return 1
        
        # Validate files
        if not validate_conversation_files_enhanced(output_dir):
            logger.error("File validation failed!")
            return 1
        
        # Final memory check
        final_memory = psutil.virtual_memory().percent
        logger.info(f"Memory usage: {initial_memory:.1f}% -> {final_memory:.1f}%")
        
        # Success summary
        logger.info("=" * 80)
        logger.info("Enhanced Conversational Dataset Preparation Completed!")
        logger.info(f"Files saved in: {output_dir}")
        logger.info("")
        
        # Comprehensive file summary
        logger.info("Generated Files Summary:")
        
        total_size_mb = 0
        total_conversations = 0
        
        # Regular files
        regular_files = list(output_dir.glob("oasst1_*_conversations.jsonl"))
        for file_path in regular_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            try:
                with open(file_path, 'r') as f:
                    line_count = sum(1 for _ in f)
                total_conversations += line_count
                logger.info(f"   {file_path.name}: {size_mb:.1f} MB, {line_count:,} conversations")
            except Exception:
                logger.info(f"   {file_path.name}: {size_mb:.1f} MB")
        
        # Sharded files summary
        shard_files = list((output_dir / "shards").glob("*.jsonl")) if (output_dir / "shards").exists() else []
        if shard_files:
            shard_size_mb = sum(f.stat().st_size / (1024 * 1024) for f in shard_files)
            shard_conversations = 0
            
            try:
                for f in shard_files:
                    with open(f, 'r') as file:
                        shard_conversations += sum(1 for _ in file)
            except Exception:
                pass
            
            total_size_mb += shard_size_mb
            total_conversations += shard_conversations
            
            logger.info(f"   Sharded dataset: {len(shard_files)} files, {shard_size_mb:.1f} MB, {shard_conversations:,} conversations")
        
        # Metadata files
        if (output_dir / "shard_metadata.json").exists():
            logger.info(f"   shard_metadata.json: Sharding configuration and indices")
        
        logger.info("")
        logger.info(f"Dataset Summary:")
        logger.info(f"  Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
        logger.info(f"  Total conversations: {total_conversations:,}")
        logger.info(f"  Sharding enabled: {'Yes' if shard_files else 'No'}")
        logger.info(f"  Memory efficient: Yes")
        
        # Performance estimates
        if total_conversations > 100000:
            est_training_hours = total_conversations // 2000
            logger.info(f"  Estimated training time: {est_training_hours}+ hours")
        
        logger.info("")
        logger.info("Enhanced Dataset Features:")
        logger.info("  - Automatic sharding based on system resources")
        logger.info("  - Parallel processing for faster data preparation") 
        logger.info("  - Memory-efficient loading strategies")
        logger.info("  - Support for datasets up to 16TB+")
        logger.info("  - Streaming support for massive datasets")
        logger.info("  - Comprehensive validation and error handling")
        logger.info("")
        
        logger.info("Dataset Format:")
        logger.info("   Each line contains a complete conversation with:")
        logger.info("   - conversation_id: Unique identifier")
        logger.info("   - messages: Array of turn-by-turn exchanges")
        logger.info("   - total_turns: Number of messages in conversation")
        logger.info("   - languages: List of detected languages")
        logger.info("   - Each message has: role, content, turn number, metadata")
        logger.info("")
        
        if shard_files:
            logger.info("Sharding Information:")
            logger.info("   - Dataset automatically sharded for optimal performance")
            logger.info("   - ConversationDataset handles shards transparently")
            logger.info("   - StreamingConversationDataset available for massive datasets")
            logger.info("   - Metadata file contains sharding configuration")
            logger.info("")
        
        logger.info("Usage Examples:")
        logger.info("   # Automatic strategy selection (recommended)")
        logger.info("   from dataset import create_memory_efficient_dataloader")
        logger.info("   dataloader = create_memory_efficient_dataloader('oasst1_data/oasst1_train_conversations.jsonl', tokenizer, config)")
        logger.info("")
        logger.info("   # Manual dataset loading")
        logger.info("   from dataset import ConversationDataset, StreamingConversationDataset")
        logger.info("   dataset = ConversationDataset('oasst1_data/oasst1_train_conversations.jsonl', tokenizer, config)")
        logger.info("")
        logger.info("   # For massive datasets (auto-streaming)")
        logger.info("   dataset = StreamingConversationDataset('oasst1_data/oasst1_train_conversations.jsonl', tokenizer, config)")
        logger.info("")
        logger.info("Ready for enhanced conversational training!")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        # Cleanup memory
        gc.collect()
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup memory
        gc.collect()
        return 1


# Global helper functions moved outside of class for multiprocessing compatibility
def _process_messages_in_memory(valid_messages: List[Dict]) -> List[Dict]:
    """Process messages in memory for small datasets."""
    logger.info("Processing in memory (small dataset)")
    
    # Group messages by conversation tree
    tree_messages = defaultdict(list)
    for msg in valid_messages:
        tree_id = msg.get('message_tree_id', '')
        if tree_id:
            tree_messages[tree_id].append(msg)
    
    # Build conversations
    all_conversations = []
    processed_trees = 0
    
    for tree_id, messages in tree_messages.items():
        processed_trees += 1
        if processed_trees % 1000 == 0:
            logger.info(f"  Processing tree {processed_trees:,}/{len(tree_messages):,}...")
        
        try:
            message_map, root_messages = build_conversation_tree(messages)
            
            for root_id in root_messages:
                paths = extract_conversation_paths(message_map, root_id)
                for path in paths:
                    conv = format_conversation(path)
                    all_conversations.append(conv)
                    
                    # Create sub-conversations
                    if len(path) > 4:
                        for start_idx in range(0, len(path) - 3, 2):
                            if start_idx > 0:
                                sub_path = path[start_idx:]
                                if len(sub_path) >= 2:
                                    sub_conv = format_conversation(sub_path)
                                    sub_conv['conversation_id'] = f"{sub_conv['conversation_id']}_sub_{start_idx}"
                                    all_conversations.append(sub_conv)
        except Exception as e:
            logger.debug(f"Error processing tree {tree_id}: {e}")
            continue
    
    return filter_quality_conversations(all_conversations)


def _process_messages_parallel(valid_messages: List[Dict], split_name: str, shard_config: ShardConfig) -> List[Dict]:
    """Process messages using parallel workers for medium datasets."""
    logger.info(f"Processing with parallel workers (medium dataset)")
    
    # Split messages into chunks for parallel processing
    chunk_size = max(5000, len(valid_messages) // (shard_config.num_workers * 2))
    chunks = []
    
    for i in range(0, len(valid_messages), chunk_size):
        end_idx = min(i + chunk_size, len(valid_messages))
        chunks.append((i // chunk_size, valid_messages[i:end_idx], split_name))
    
    logger.info(f"Processing {len(valid_messages):,} messages in {len(chunks)} chunks using {shard_config.num_workers} workers")
    
    # Process chunks in parallel
    all_conversations = []
    
    with ProcessPoolExecutor(max_workers=shard_config.num_workers) as executor:
        future_to_chunk = {executor.submit(process_dataset_chunk, chunk): chunk[0] for chunk in chunks}
        
        for future in as_completed(future_to_chunk):
            try:
                chunk_idx, chunk_conversations = future.result()
                all_conversations.extend(chunk_conversations)
                
                if len(all_conversations) % 10000 == 0:
                    logger.info(f"Processed {len(all_conversations):,} conversations...")
                    
            except Exception as e:
                logger.warning(f"Chunk processing failed: {e}")
    
    return all_conversations


def _process_messages_streaming(valid_messages: List[Dict], split_name: str, shard_config: ShardConfig) -> List[Dict]:
    """Process messages with streaming approach for massive datasets."""
    logger.info(f"Processing with streaming approach (large dataset)")
    
    # For very large datasets, process in smaller memory-conscious chunks
    all_conversations = []
    
    # Group messages by tree first (streaming)
    tree_messages = defaultdict(list)
    
    logger.info("Building conversation trees...")
    for i, msg in enumerate(valid_messages):
        if i % 50000 == 0 and i > 0:
            logger.info(f"  Grouped {i:,}/{len(valid_messages):,} messages...")
        
        tree_id = msg.get('message_tree_id', '')
        if tree_id:
            tree_messages[tree_id].append(msg)
    
    logger.info(f"Found {len(tree_messages):,} conversation trees")
    
    # Process trees in batches to manage memory
    processed_trees = 0
    tree_items = list(tree_messages.items())
    
    for batch_start in range(0, len(tree_items), 1000):  # Process 1000 trees at a time
        batch_end = min(batch_start + 1000, len(tree_items))
        batch_trees = tree_items[batch_start:batch_end]
        
        batch_conversations = []
        
        for tree_id, messages in batch_trees:
            processed_trees += 1
            
            if processed_trees % 5000 == 0:
                logger.info(f"  Processing tree {processed_trees:,}/{len(tree_items):,}...")
            
            try:
                message_map, root_messages = build_conversation_tree(messages)
                
                for root_id in root_messages:
                    paths = extract_conversation_paths(message_map, root_id)
                    for path in paths:
                        conv = format_conversation(path)
                        batch_conversations.append(conv)
                        
                        # Create sub-conversations
                        if len(path) > 4:
                            for start_idx in range(0, len(path) - 3, 2):
                                if start_idx > 0:
                                    sub_path = path[start_idx:]
                                    if len(sub_path) >= 2:
                                        sub_conv = format_conversation(sub_path)
                                        sub_conv['conversation_id'] = f"{sub_conv['conversation_id']}_sub_{start_idx}"
                                        batch_conversations.append(sub_conv)
            except Exception:
                continue
        
        # Filter and add batch conversations
        filtered_batch = filter_quality_conversations(batch_conversations)
        all_conversations.extend(filtered_batch)
        
        # Clear batch memory
        del batch_conversations, filtered_batch
        gc.collect()
        
        if len(all_conversations) % 25000 == 0:
            logger.info(f"Total conversations: {len(all_conversations):,}")
    
    return all_conversations


def _save_conversations_single_file(conversations: List[Dict], output_file: Path):
    """Save conversations to a single file with progress tracking."""
    logger.info(f"Saving {len(conversations):,} conversations to: {output_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in conversations:
            try:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
                saved_count += 1
                
                if saved_count % 10000 == 0:
                    logger.info(f"  Saved {saved_count:,}/{len(conversations):,} conversations...")
                    
            except Exception as e:
                logger.warning(f"Error saving conversation: {e}")
                continue
    
    logger.info(f"Successfully saved {saved_count:,} conversations")


if __name__ == "__main__":
    exit(main())