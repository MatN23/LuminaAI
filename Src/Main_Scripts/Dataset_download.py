# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import logging
import json
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Install with: pip install datasets")
    print("Run: pip install datasets huggingface_hub")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_output_directory(project_root: Optional[str] = None) -> Path:
    """Setup and create output directory for dataset files."""
    if project_root is None:
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir
    
    output_dir = Path(project_root) / "oasst1_data"
    
    # Ensure the directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created output directory: {output_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to create output directory: {e}")
        raise
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {output_dir}")
    
    return output_dir

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
        if conv['messages'][0]['role'] != 'prompter':
            continue
            
        # Must alternate between prompter and assistant
        expected_role = 'prompter'
        for msg in conv['messages']:
            if msg['role'] != expected_role:
                valid = False
                break
            expected_role = 'assistant' if expected_role == 'prompter' else 'prompter'
        
        if not valid:
            continue
        
        # Very relaxed filtering to maximize dataset size
        has_content = True
        for msg in conv['messages']:
            # Only filter out completely empty messages
            if not msg['content'] or len(msg['content'].strip()) == 0:
                has_content = False
                break
                
        if not has_content:
            continue
            
        quality_conversations.append(conv)
    
    return quality_conversations

def analyze_conversations(conversations: List[Dict], split_name: str):
    """Analyze the structure and quality of conversations."""
    if not conversations:
        logger.warning(f"No conversations found in {split_name}")
        return
    
    total_conversations = len(conversations)
    total_turns = sum(len(conv['messages']) for conv in conversations)
    avg_turns = total_turns / total_conversations if total_conversations > 0 else 0
    
    turn_distribution = defaultdict(int)
    role_counts = defaultdict(int)
    
    for conv in conversations:
        turn_distribution[len(conv['messages'])] += 1
        for msg in conv['messages']:
            role_counts[msg['role']] += 1
    
    logger.info(f"📊 Conversation Analysis for {split_name}:")
    logger.info(f"  Total conversations: {total_conversations:,}")
    logger.info(f"  Total turns: {total_turns:,}")
    logger.info(f"  Average turns per conversation: {avg_turns:.1f}")
    logger.info(f"  Turn distribution (top 10):")
    
    for turns, count in sorted(turn_distribution.items())[:10]:
        logger.info(f"    {turns} turns: {count:,} conversations ({count/total_conversations*100:.1f}%)")
    
    logger.info(f"  Role distribution:")
    for role, count in role_counts.items():
        logger.info(f"    {role}: {count:,} messages ({count/total_turns*100:.1f}%)")

def download_and_process_conversations(output_dir: Path) -> bool:
    """Download OASST1 dataset and process into conversation format."""
    try:
        logger.info("📦 Loading OpenAssistant dataset (oasst1)...")
        logger.info("This may take a few minutes for the first download...")
        
        # Load dataset with error handling
        try:
            ds = load_dataset("OpenAssistant/oasst1", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.info("💡 Possible solutions:")
            logger.info("  1. Try running: huggingface-cli login")
            logger.info("  2. Install git-lfs if not installed: sudo apt-get install git-lfs")
            logger.info("  3. Check your internet connection")
            return False
        
        logger.info("✅ Dataset loaded successfully!")
        logger.info(f"📊 Available splits: {list(ds.keys())}")
        
        # Process train, validation, and evaluation splits
        splits_to_process = ['train', 'validation']
        
        # Check if evaluation split exists and add it
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
                
            logger.info(f"📄 Processing {split_name} split...")
            
            # Get split data
            split_data = ds[split_name]
            logger.info(f"📊 Total messages in {split_name}: {len(split_data):,}")
            
            # Filter for messages - very relaxed filtering to maximize data
            valid_messages = []
            
            for i, msg in enumerate(split_data):
                # Show progress for large datasets
                if i > 0 and i % 10000 == 0:
                    logger.info(f"  Processed {i:,}/{len(split_data):,} messages...")
                
                # Very basic validation - keep as much data as possible
                if (msg.get('text', '').strip() and  # Has some content
                    msg.get('message_tree_id')):  # Has tree ID
                    valid_messages.append(msg)
            
            logger.info(f"📊 Valid messages in {split_name}: {len(valid_messages):,}")
            
            # Group messages by conversation tree
            tree_messages = defaultdict(list)
            for msg in valid_messages:
                tree_id = msg.get('message_tree_id', '')
                if tree_id:
                    tree_messages[tree_id].append(msg)
            
            logger.info(f"📊 Conversation trees in {split_name}: {len(tree_messages):,}")
            
            # Build conversations - extract more paths per tree
            all_conversations = []
            processed_trees = 0
            
            for tree_id, messages in tree_messages.items():
                processed_trees += 1
                if processed_trees % 1000 == 0:
                    logger.info(f"  Processing tree {processed_trees:,}/{len(tree_messages):,}...")
                
                # Build conversation tree
                message_map, root_messages = build_conversation_tree(messages)
                
                # Extract conversation paths from each root
                for root_id in root_messages:
                    paths = extract_conversation_paths(message_map, root_id)
                    for path in paths:
                        conv = format_conversation(path)
                        all_conversations.append(conv)
                        
                        # Also create sub-conversations from longer paths to maximize data
                        if len(path) > 4:  # If conversation is long enough
                            # Create sub-conversations starting from different points
                            for start_idx in range(0, len(path) - 3, 2):  # Every 2 messages
                                if start_idx > 0:
                                    sub_path = path[start_idx:]
                                    if len(sub_path) >= 2:  # Ensure minimum length
                                        sub_conv = format_conversation(sub_path)
                                        # Add suffix to make unique ID
                                        sub_conv['conversation_id'] = f"{sub_conv['conversation_id']}_sub_{start_idx}"
                                        all_conversations.append(sub_conv)
            
            logger.info(f"📊 Raw conversations extracted from {split_name}: {len(all_conversations):,}")
            
            # Apply filtering
            filtered_conversations = filter_quality_conversations(all_conversations, strict_filtering=False)
            logger.info(f"📊 Filtered conversations: {len(filtered_conversations):,}")
            
            # Analyze dataset
            logger.info(f"\n--- Analysis for {split_name} ---")
            analyze_conversations(filtered_conversations, split_name)
            
            # Save main conversation file
            if not filtered_conversations:
                logger.warning(f"No conversations to save for {split_name}")
                continue
                
            output_file = output_dir / f"oasst1_{split_name}_conversations.jsonl"
            logger.info(f"💾 Saving {len(filtered_conversations):,} {split_name} conversations to: {output_file}")
            
            # Ensure parent directory exists before writing
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            saved_count = 0
            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in filtered_conversations:
                    try:
                        f.write(json.dumps(conv, ensure_ascii=False) + '\n')
                        saved_count += 1
                        
                        if saved_count % 5000 == 0:
                            logger.info(f"  Saved {saved_count:,}/{len(filtered_conversations):,} conversations...")
                            
                    except Exception as e:
                        logger.warning(f"Error saving conversation: {e}")
                        continue
            
            logger.info(f"✅ Saved {saved_count:,} {split_name} conversations")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing conversations: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_conversation_files(output_dir: Path) -> bool:
    """Validate that conversation files exist and are readable."""
    # Get all conversation files in the directory
    conversation_files = list(output_dir.glob("oasst1_*_conversations.jsonl"))
    
    if not conversation_files:
        logger.error("❌ No conversation files found")
        return False
    
    for file_path in conversation_files:
        if not file_path.exists():
            logger.error(f"❌ Missing file: {file_path}")
            return False
        
        if file_path.stat().st_size == 0:
            logger.error(f"❌ Empty file: {file_path}")
            return False
        
        # Test reading and parsing
        try:
            line_count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line_count += 1
                    
                    # Test parsing of first few lines
                    if i < 3:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            # Check conversation structure
                            required_fields = ['conversation_id', 'messages', 'total_turns']
                            for field in required_fields:
                                if field not in data:
                                    logger.error(f"❌ Missing field '{field}' in {file_path}")
                                    return False
                            
                            # Check message structure
                            if data['messages']:
                                msg = data['messages'][0]
                                msg_fields = ['role', 'content', 'turn']
                                for field in msg_fields:
                                    if field not in msg:
                                        logger.error(f"❌ Missing message field '{field}' in {file_path}")
                                        return False
                                        
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ Invalid JSON in {file_path} line {i+1}: {e}")
                            return False
            
            logger.info(f"✅ {file_path.name}: {line_count:,} conversations validated")
                        
        except Exception as e:
            logger.error(f"❌ Cannot read file {file_path}: {e}")
            return False
    
    logger.info("✅ All conversation files validated successfully!")
    return True

def check_existing_files(output_dir: Path) -> bool:
    """Check if dataset files already exist and are valid."""
    # Look for any existing conversation files
    existing_files = list(output_dir.glob("oasst1_*_conversations.jsonl"))
    
    if existing_files:
        logger.info("🔍 Found existing conversation files!")
        for file_path in existing_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  - {file_path.name}: {size_mb:.1f} MB")
        
        logger.info("🔄 Validating existing files...")
        return validate_conversation_files(output_dir)
    
    return False

def main():
    """Main function to download and prepare OASST1 conversational dataset."""
    logger.info("🚀 Starting OASST1 Conversational Dataset Download...")
    logger.info("=" * 70)
    
    try:
        # Setup directories
        output_dir = setup_output_directory()
        
        # Check if files already exist and are valid
        if check_existing_files(output_dir):
            logger.info("✅ Valid dataset files already exist!")
            logger.info("💡 Delete files in the output directory if you want to re-download")
            logger.info("=" * 70)
            return 0
        else:
            logger.info("🔄 Downloading/reprocessing dataset...")
        
        # Download and process conversations
        success = download_and_process_conversations(output_dir)
        
        if not success:
            logger.error("❌ Dataset processing failed!")
            return 1
        
        # Validate files
        if not validate_conversation_files(output_dir):
            logger.error("❌ File validation failed!")
            return 1
        
        # Success summary
        logger.info("=" * 70)
        logger.info("🎉 Conversational dataset preparation completed!")
        logger.info(f"📁 Files saved in: {output_dir}")
        logger.info("")
        logger.info("📋 Generated Files:")
        
        # List all generated files with sizes
        for file_path in sorted(output_dir.glob("*.jsonl")):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            line_count = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
            logger.info(f"   ✅ {file_path.name}: {size_mb:.1f} MB, {line_count:,} conversations")
        
        logger.info("")
        logger.info("📋 Dataset Format:")
        logger.info("   Each line contains a complete conversation with:")
        logger.info("   - conversation_id: Unique identifier")
        logger.info("   - messages: Array of turn-by-turn exchanges")
        logger.info("   - total_turns: Number of messages in conversation")
        logger.info("   - Each message has: role, content, turn number")
        logger.info("")
        logger.info("🚀 Ready for conversational training!")
        logger.info("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("ℹ️ Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())