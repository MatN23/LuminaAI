# Copywrite (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

from datasets import load_dataset
import os

def main():
    print("🚀 Running dataset download script...")

    # Get absolute path of script folder
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Assuming your project root is 3 levels up from script (adjust as needed)
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    print("Detected project root:", project_root)

    # Desired output directory inside project
    output_dir = os.path.join(project_root, "main", "src", "oasst1_data")
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print("📦 Loading OpenAssistant dataset (oasst1)...")
    ds = load_dataset("OpenAssistant/oasst1")
    print("✅ Dataset loaded!")

    print(f"📊 Full train split size: {len(ds['train'])}")
    print(f"📊 Full validation split size: {len(ds['validation'])}")

    # Limit train split size to 20,000 examples (adjust as needed)
    max_train_examples = 20000
    train_subset = ds["train"].select(range(min(max_train_examples, len(ds["train"]))))
    print(f"📊 Using train subset size: {len(train_subset)}")

    # Save subset train split
    train_path = os.path.join(output_dir, "oasst1_train.jsonl")
    print(f"💾 Saving train subset to: {train_path}")
    train_subset.to_json(train_path, orient="records", lines=True)

    # Save full validation split (usually smaller, but can also subset if needed)
    val_path = os.path.join(output_dir, "oasst1_validation.jsonl")
    print(f"💾 Saving validation split to: {val_path}")
    ds["validation"].to_json(val_path, orient="records", lines=True)

    print("✅ Dataset saved successfully.")

if __name__ == "__main__":
    main()
