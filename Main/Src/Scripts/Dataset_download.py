# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import os
import logging
from typing import Optional
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Install with: pip install datasets")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_output_directory(project_root: Optional[str] = None) -> Path:
    """Setup and create output directory for dataset files."""
    if project_root is None:
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.parent
    
    output_dir = Path(project_root) / "oasst1_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {output_dir}")
    
    return output_dir

def download_and_save_dataset(output_dir: Path, max_train_examples: int = 20000) -> bool:
    """Download OASST1 dataset and save to JSONL files."""
    try:
        logger.info("📦 Loading OpenAssistant dataset (oasst1)...")
        ds = load_dataset("OpenAssistant/oasst1")
        logger.info("✅ Dataset loaded successfully!")
        
        # Log dataset information
        train_size = len(ds['train'])
        val_size = len(ds['validation'])
        logger.info(f"📊 Full train split size: {train_size:,}")
        logger.info(f"📊 Full validation split size: {val_size:,}")
        
        # Create train subset if needed
        actual_train_size = min(max_train_examples, train_size)
        if actual_train_size < train_size:
            train_subset = ds["train"].select(range(actual_train_size))
            logger.info(f"📊 Using train subset size: {actual_train_size:,}")
        else:
            train_subset = ds["train"]
            logger.info(f"📊 Using full train split: {actual_train_size:,}")
        
        # Save train split
        train_path = output_dir / "oasst1_train.jsonl"
        logger.info(f"💾 Saving train data to: {train_path}")
        train_subset.to_json(str(train_path), orient="records", lines=True)
        
        # Save validation split
        val_path = output_dir / "oasst1_validation.jsonl"
        logger.info(f"💾 Saving validation data to: {val_path}")
        ds["validation"].to_json(str(val_path), orient="records", lines=True)
        
        # Verify files were created
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError("Dataset files were not created successfully")
        
        logger.info("✅ Dataset saved successfully!")
        logger.info(f"   Train file: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
        logger.info(f"   Validation file: {val_path} ({val_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading dataset: {e}")
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
        
        # Test reading first line
        try:
            with open(path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if not first_line.strip():
                    logger.error(f"❌ Invalid JSONL format: {path}")
                    return False
        except Exception as e:
            logger.error(f"❌ Cannot read file {path}: {e}")
            return False
    
    logger.info("✅ Dataset files validated successfully!")
    return True

def main():
    """Main function to download and prepare OASST1 dataset."""
    logger.info("🚀 Starting dataset download script...")
    
    try:
        # Setup directories
        output_dir = setup_output_directory()
        
        # Download and save dataset
        max_train_examples = 20000  # Adjust as needed
        success = download_and_save_dataset(output_dir, max_train_examples)
        
        if not success:
            logger.error("❌ Dataset download failed!")
            return 1
        
        # Validate files
        if not validate_dataset_files(output_dir):
            logger.error("❌ Dataset validation failed!")
            return 1
        
        logger.info("🎉 Dataset preparation completed successfully!")
        logger.info(f"📁 Files saved in: {output_dir}")
        logger.info("   - oasst1_train.jsonl")
        logger.info("   - oasst1_validation.jsonl")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Download interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())