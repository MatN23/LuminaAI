from datasets import load_dataset
import os

print("🚀 Running dataset download script...")

# --- Load dataset from Hugging Face ---
print("📦 Loading OpenAssistant dataset (oasst1)...")
ds = load_dataset("OpenAssistant/oasst1")
print("✅ Dataset loaded!")

# --- Print basic info ---
print(f"📊 Train split: {len(ds['train'])} examples")
print(f"📊 Validation split: {len(ds['validation'])} examples")

# --- Define output directory ---
output_dir = "oasst1_data"
os.makedirs(output_dir, exist_ok=True)

# --- Save splits to JSONL format ---
train_path = os.path.join(output_dir, "oasst1_train.jsonl")
val_path = os.path.join(output_dir, "oasst1_validation.jsonl")

print(f"💾 Saving train split to: {train_path}")
ds["train"].to_json(train_path, orient="records", lines=True)

print(f"💾 Saving validation split to: {val_path}")
ds["validation"].to_json(val_path, orient="records", lines=True)

print("✅ All done! Dataset saved successfully.")
