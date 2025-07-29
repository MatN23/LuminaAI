# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import logging
import json
from typing import Optional
from pathlib import Path

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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {output_dir}")
    
    return output_dir

def analyze_dataset_sample(dataset, split_name: str, num_samples: int = 5):
    """Analyze a few samples from the dataset to understand structure."""
    logger.info(f"Analyzing {split_name} split structure...")
    
    sample_data = dataset.select(range(min(num_samples, len(dataset))))
    
    for i, example in enumerate(sample_data):
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Role: {example.get('role', 'N/A')}")
        logger.info(f"  Language: {example.get('lang', 'N/A')}")
        logger.info(f"  Text length: {len(example.get('text', ''))}")
        logger.info(f"  Deleted: {example.get('deleted', 'N/A')}")
        logger.info(f"  Text preview: {example.get('text', '')[:100]}...")
        logger.info("-" * 40)

def validate_dataset_quality(dataset, split_name: str):
    """Validate the quality and structure of the loaded dataset."""
    logger.info(f"Validating {split_name} dataset quality...")
    
    total_samples = len(dataset)
    stats = {
        'total': total_samples,
        'deleted': 0,
        'english': 0,
        'non_english': 0,
        'empty_text': 0,
        'by_role': {},
        'languages': set()
    }
    
    # Sample a subset for analysis if dataset is large
    sample_size = min(10000, total_samples)
    sample_indices = range(0, total_samples, max(1, total_samples // sample_size))
    
    for i in sample_indices:
        if i >= total_samples:
            break
            
        example = dataset[i]
        
        # Count deleted
        if example.get('deleted', False):
            stats['deleted'] += 1
        
        # Count languages
        lang = example.get('lang', 'unknown')
        stats['languages'].add(lang)
        if lang == 'en':
            stats['english'] += 1
        else:
            stats['non_english'] += 1
        
        # Count by role
        role = example.get('role', 'unknown')
        stats['by_role'][role] = stats['by_role'].get(role, 0) + 1
        
        # Count empty text
        if not example.get('text', '').strip():
            stats['empty_text'] += 1
    
    # Scale up the counts based on sampling ratio
    scale_factor = total_samples / len(list(sample_indices))
    for key in ['deleted', 'english', 'non_english', 'empty_text']:
        stats[key] = int(stats[key] * scale_factor)
    
    for role in stats['by_role']:
        stats['by_role'][role] = int(stats['by_role'][role] * scale_factor)
    
    # Log statistics
    logger.info(f"Dataset Quality Report for {split_name}:")
    logger.info(f"  Total samples: {stats['total']:,}")
    logger.info(f"  Deleted samples: {stats['deleted']:,} ({stats['deleted']/stats['total']*100:.1f}%)")
    logger.info(f"  English samples: {stats['english']:,} ({stats['english']/stats['total']*100:.1f}%)")
    logger.info(f"  Non-English samples: {stats['non_english']:,}")
    logger.info(f"  Empty text samples: {stats['empty_text']:,}")
    logger.info(f"  Languages found: {len(stats['languages'])} ({', '.join(sorted(list(stats['languages']))[:10])})")
    logger.info(f"  By role:")
    for role, count in stats['by_role'].items():
        logger.info(f"    {role}: {count:,} ({count/stats['total']*100:.1f}%)")
    
    return stats

def download_and_save_dataset(output_dir: Path, max_train_examples: int = 50000) -> bool:
    """Download OASST1 dataset and save to JSONL files with enhanced processing."""
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
        
        # Log dataset information
        train_size = len(ds['train'])
        val_size = len(ds['validation'])
        logger.info(f"📊 Full train split size: {train_size:,}")
        logger.info(f"📊 Full validation split size: {val_size:,}")
        
        # Analyze dataset structure
        analyze_dataset_sample(ds['train'], 'train')
        train_stats = validate_dataset_quality(ds['train'], 'train')
        
        # Create train subset if needed
        actual_train_size = min(max_train_examples, train_size)
        if actual_train_size < train_size:
            logger.info(f"📊 Using train subset size: {actual_train_size:,}")
            train_subset = ds["train"].select(range(actual_train_size))
        else:
            logger.info(f"📊 Using full train split: {actual_train_size:,}")
            train_subset = ds["train"]
        
        # Save train split with enhanced processing
        train_path = output_dir / "oasst1_train.jsonl"
        logger.info(f"💾 Saving train data to: {train_path}")
        
        processed_count = 0
        skipped_count = 0
        
        with open(train_path, 'w', encoding='utf-8') as f:
            for i, example in enumerate(train_subset):
                try:
                    # Clean and validate the example
                    clean_example = {
                        'message_id': str(example.get('message_id', '')),
                        'parent_id': str(example.get('parent_id', '')),  
                        'user_id': str(example.get('user_id', '')),
                        'created_date': str(example.get('created_date', '')),
                        'text': str(example.get('text', '')).strip(),
                        'role': str(example.get('role', '')).lower(),
                        'lang': str(example.get('lang', '')),
                        'review_count': int(example.get('review_count', 0)),
                        'review_result': bool(example.get('review_result', False)),
                        'deleted': bool(example.get('deleted', False)),
                        'rank': int(example.get('rank', 0)),
                        'synthetic': bool(example.get('synthetic', False)),
                        'model_name': str(example.get('model_name', '')),
                        'message_tree_id': str(example.get('message_tree_id', '')),
                        'tree_state': str(example.get('tree_state', ''))
                    }
                    
                    # Skip if essential fields are missing
                    if not clean_example['text'] or not clean_example['role']:
                        skipped_count += 1
                        continue
                    
                    f.write(json.dumps(clean_example, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                    if processed_count % 5000 == 0:
                        logger.info(f"Processed {processed_count:,} train samples...")
                        
                except Exception as e:
                    logger.warning(f"Skipping malformed train sample {i}: {e}")
                    skipped_count += 1
                    continue
        
        logger.info(f"Train processing complete: {processed_count:,} saved, {skipped_count:,} skipped")
        
        # Save validation split
        val_path = output_dir / "oasst1_validation.jsonl"
        logger.info(f"💾 Saving validation data to: {val_path}")
        
        val_processed = 0
        val_skipped = 0
        
        with open(val_path, 'w', encoding='utf-8') as f:
            for i, example in enumerate(ds["validation"]):
                try:
                    clean_example = {
                        'message_id': str(example.get('message_id', '')),
                        'parent_id': str(example.get('parent_id', '')),
                        'user_id': str(example.get('user_id', '')),
                        'created_date': str(example.get('created_date', '')),
                        'text': str(example.get('text', '')).strip(),
                        'role': str(example.get('role', '')).lower(),
                        'lang': str(example.get('lang', '')),
                        'review_count': int(example.get('review_count', 0)),
                        'review_result': bool(example.get('review_result', False)),
                        'deleted': bool(example.get('deleted', False)),
                        'rank': int(example.get('rank', 0)),
                        'synthetic': bool(example.get('synthetic', False)),
                        'model_name': str(example.get('model_name', '')),
                        'message_tree_id': str(example.get('message_tree_id', '')),
                        'tree_state': str(example.get('tree_state', ''))
                    }
                    
                    if not clean_example['text'] or not clean_example['role']:
                        val_skipped += 1
                        continue
                    
                    f.write(json.dumps(clean_example, ensure_ascii=False) + '\n')
                    val_processed += 1
                    
                except Exception as e:
                    logger.warning(f"Skipping malformed validation sample {i}: {e}")
                    val_skipped += 1
                    continue
        
        logger.info(f"Validation processing complete: {val_processed:,} saved, {val_skipped:,} skipped")
        
        # Verify files were created
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError("Dataset files were not created successfully")
        
        # Final statistics
        train_size_mb = train_path.stat().st_size / 1024 / 1024
        val_size_mb = val_path.stat().st_size / 1024 / 1024
        
        logger.info("✅ Dataset saved successfully!")
        logger.info(f"   Train file: {train_path} ({train_size_mb:.1f} MB, {processed_count:,} samples)")
        logger.info(f"   Validation file: {val_path} ({val_size_mb:.1f} MB, {val_processed:,} samples)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_dataset_files(output_dir: Path) -> bool:
    """Validate that dataset files exist and are readable."""
    train_path = output_dir / "oasst1_train.jsonl"
    val_path = output_dir / "oasst1_validation.jsonl"
    
    for path in [train_path, val_path]:
        if not path.exists():
            logger.error(f"❌ Missing file: {path}")
            return False
        
        if path.stat().st_size == 0:
            logger.error(f"❌ Empty file: {path}")
            return False
        
        # Test reading and parsing first few lines
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 3:  # Test first 3 lines
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Try to parse as JSON
                    try:
                        data = json.loads(line)
                        # Check required fields
                        required_fields = ['text', 'role', 'lang']
                        for field in required_fields:
                            if field not in data:
                                logger.error(f"❌ Missing required field '{field}' in {path}")
                                return False
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Invalid JSON in {path}: {e}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Cannot read file {path}: {e}")
            return False
    
    logger.info("✅ Dataset files validated successfully!")
    return True

def main():
    """Main function to download and prepare OASST1 dataset."""
    logger.info("🚀 Starting enhanced OASST1 dataset download...")
    logger.info("=" * 60)
    
    try:
        # Setup directories
        output_dir = setup_output_directory()
        
        # Check if files already exist
        train_path = output_dir / "oasst1_train.jsonl"
        val_path = output_dir / "oasst1_validation.jsonl"
        
        if train_path.exists() and val_path.exists():
            logger.info("📁 Dataset files already exist!")
            logger.info("Delete them if you want to re-download:")
            logger.info(f"  - {train_path}")
            logger.info(f"  - {val_path}")
            
            # Validate existing files
            if validate_dataset_files(output_dir):
                logger.info("✅ Existing files are valid!")
                return 0
            else:
                logger.info("⚠️  Existing files are invalid, re-downloading...")
        
        # Download and save dataset
        max_train_examples = 100000  # Adjust based on your needs
        logger.info(f"📊 Maximum training samples: {max_train_examples:,}")
        
        success = download_and_save_dataset(output_dir, max_train_examples)
        
        if not success:
            logger.error("❌ Dataset download failed!")
            return 1
        
        # Validate files
        if not validate_dataset_files(output_dir):
            logger.error("❌ Dataset validation failed!")
            return 1
        
        # Success summary
        logger.info("=" * 60)
        logger.info("🎉 Dataset preparation completed successfully!")
        logger.info(f"📁 Files saved in: {output_dir}")
        logger.info("   ✅ oasst1_train.jsonl")
        logger.info("   ✅ oasst1_validation.jsonl")
        logger.info("")
        logger.info("🚀 Ready to start training! Run: python Train.py")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Download interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())