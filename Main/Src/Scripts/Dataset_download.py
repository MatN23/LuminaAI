from datasets import load_dataset
import os

print("Running dataset download script...")

# Load dataset
print("Loading OpenAssistant dataset...")
ds = load_dataset("OpenAssistant/oasst1")
print("Dataset loaded!")

# Print number of examples
print(f"Train split: {len(ds['train'])} examples")
print(f"Validation split: {len(ds['validation'])} examples")

# Define output directory
output_dir = "oasst1_data"
os.makedirs(output_dir, exist_ok=True)

# Save to JSONL
train_path = os.path.join(output_dir, "oasst1_train.jsonl")
val_path = os.path.join(output_dir, "oasst1_validation.jsonl")

print(f"Saving train split to: {train_path}")
ds["train"].to_json(train_path, orient="records", lines=True)

print(f"Saving validation split to: {val_path}")
ds["validation"].to_json(val_path, orient="records", lines=True)

print("✅ Dataset saved successfully!")
