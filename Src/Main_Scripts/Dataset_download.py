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

def build_conversation_tree(messages: List[Dict]) -> Dict[str, Dict]:
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
        
        # If this is a leaf node or has no valid children, save the conversation
        if not node['children']:
            if len(new_path) >= 2:  # At least one exchange (user + assistant)
                conversations.append(new_path)
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
            'review_result': msg.get('review_result', False),
            'rank': msg.get('rank', 0),
            'synthetic': msg.get('synthetic', False),
            'model_name': msg.get('model_name', '')
        }
        conversation['messages'].append(formatted_msg)
    
    return conversation

def filter_quality_conversations(conversations: List[Dict]) -> List[Dict]:
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
            
        # Filter out very short or low-quality messages
        has_substantial_content = True
        for msg in conv['messages']:
            if len(msg['content']) < 10:  # Very short messages
                has_substantial_content = False
                break
                
        if not has_substantial_content:
            continue
            
        # Filter out conversations with poor review results
        if any(msg.get('review_result') == False for msg in conv['messages']):
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
    
    logger.info(f"Conversation Analysis for {split_name}:")
    logger.info(f"  Total conversations: {total_conversations:,}")
    logger.info(f"  Total turns: {total_turns:,}")
    logger.info(f"  Average turns per conversation: {avg_turns:.1f}")
    logger.info(f"  Turn distribution (top 5):")
    
    for turns, count in sorted(turn_distribution.items())[:5]:
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
            logger.info("Try running: huggingface-cli login")
            return False
        
        logger.info("✅ Dataset loaded successfully!")
        
        # Process each split
        for split_name in ['train', 'validation']:
            if split_name not in ds:
                logger.warning(f"Split '{split_name}' not found in dataset")
                continue
                
            logger.info(f"🔍 Processing {split_name} split...")
            
            # Filter for English and non-deleted messages
            split_data = ds[split_name]
            english_messages = []
            
            for msg in split_data:
                if (msg.get('lang') == 'en' and 
                    not msg.get('deleted', False) and 
                    msg.get('text', '').strip()):
                    english_messages.append(msg)
            
            logger.info(f"📊 English messages in {split_name}: {len(english_messages):,}")
            
            # Group messages by conversation tree
            tree_messages = defaultdict(list)
            for msg in english_messages:
                tree_id = msg.get('message_tree_id', '')
                if tree_id:
                    tree_messages[tree_id].append(msg)
            
            logger.info(f"📊 Conversation trees in {split_name}: {len(tree_messages):,}")
            
            # Build conversations
            all_conversations = []
            
            for tree_id, messages in tree_messages.items():
                # Build conversation tree
                message_map, root_messages = build_conversation_tree(messages)
                
                # Extract conversation paths from each root
                for root_id in root_messages:
                    paths = extract_conversation_paths(message_map, root_id)
                    for path in paths:
                        conv = format_conversation(path)
                        all_conversations.append(conv)
            
            logger.info(f"📊 Raw conversations extracted from {split_name}: {len(all_conversations):,}")
            
            # Filter for quality
            quality_conversations = filter_quality_conversations(all_conversations)
            logger.info(f"📊 Quality conversations in {split_name}: {len(quality_conversations):,}")
            
            # Analyze conversations
            analyze_conversations(quality_conversations, split_name)
            
            # Save conversations
            output_file = output_dir / f"oasst1_{split_name}_conversations.jsonl"
            logger.info(f"💾 Saving {split_name} conversations to: {output_file}")
            
            # Ensure parent directory exists before writing
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            saved_count = 0
            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in quality_conversations:
                    try:
                        f.write(json.dumps(conv, ensure_ascii=False) + '\n')
                        saved_count += 1
                        
                        if saved_count % 1000 == 0:
                            logger.info(f"Saved {saved_count:,} conversations...")
                            
                    except Exception as e:
                        logger.warning(f"Error saving conversation: {e}")
                        continue
            
            logger.info(f"✅ Saved {saved_count:,} {split_name} conversations")
            
            # Save a sample for inspection
            sample_file = output_dir / f"oasst1_{split_name}_sample.json"
            if quality_conversations:
                # Ensure directory exists
                sample_file.parent.mkdir(parents=True, exist_ok=True)
                with open(sample_file, 'w', encoding='utf-8') as f:
                    sample = quality_conversations[:5]  # First 5 conversations
                    json.dump(sample, f, ensure_ascii=False, indent=2)
                logger.info(f"📝 Sample conversations saved to: {sample_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing conversations: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_conversation_files(output_dir: Path) -> bool:
    """Validate that conversation files exist and are readable."""
    for split in ['train', 'validation']:
        file_path = output_dir / f"oasst1_{split}_conversations.jsonl"
        
        if not file_path.exists():
            logger.error(f"❌ Missing file: {file_path}")
            return False
        
        if file_path.stat().st_size == 0:
            logger.error(f"❌ Empty file: {file_path}")
            return False
        
        # Test reading and parsing
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 3:  # Test first 3 lines
                        break
                    
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
                        logger.error(f"❌ Invalid JSON in {file_path}: {e}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Cannot read file {file_path}: {e}")
            return False
    
    logger.info("✅ Conversation files validated successfully!")
    return True

def main():
    """Main function to download and prepare OASST1 conversational dataset."""
    logger.info("🚀 Starting OASST1 Conversational Dataset Download...")
    logger.info("=" * 60)
    
    try:
        # Setup directories
        output_dir = setup_output_directory()
        
        # Check if files already exist
        train_file = output_dir / "oasst1_train_conversations.jsonl"
        val_file = output_dir / "oasst1_validation_conversations.jsonl"
        
        if train_file.exists() and val_file.exists():
            logger.info("📁 Conversation files already exist!")
            logger.info("Delete them if you want to re-download:")
            logger.info(f"  - {train_file}")
            logger.info(f"  - {val_file}")
            
            # Validate existing files
            if validate_conversation_files(output_dir):
                logger.info("✅ Existing files are valid!")
                return 0
            else:
                logger.info("⚠️  Existing files are invalid, re-downloading...")
        
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
        logger.info("=" * 60)
        logger.info("🎉 Conversational dataset preparation completed!")
        logger.info(f"📁 Files saved in: {output_dir}")
        logger.info("   ✅ oasst1_train_conversations.jsonl")
        logger.info("   ✅ oasst1_validation_conversations.jsonl")
        logger.info("   ✅ oasst1_train_sample.json (for inspection)")
        logger.info("   ✅ oasst1_validation_sample.json (for inspection)")
        logger.info("")
        logger.info("📋 Dataset Format:")
        logger.info("   Each line contains a complete conversation with:")
        logger.info("   - conversation_id: Unique identifier")
        logger.info("   - messages: Array of turn-by-turn exchanges")
        logger.info("   - total_turns: Number of messages in conversation")
        logger.info("   - Each message has: role, content, turn number")
        logger.info("")
        logger.info("🚀 Ready for conversational training!")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())