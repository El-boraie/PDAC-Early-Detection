from huggingface_hub import snapshot_download
import os

output_dir = "./data/raw/NIH_Pancreas_CT"   # creates a 'data' folder in C:\FYP
os.makedirs(output_dir, exist_ok=True)

print("Starting download of NIH Pancreas-CT dataset (~7-8GB)...")
print(f"Saving to: {os.path.abspath(output_dir)}")

path = snapshot_download(
    repo_id="huggingface/CADS-dataset",
    repo_type="dataset",
    local_dir=output_dir,
    allow_patterns="0024_pancreas_ct/*",   # only the pancreas-CT folder
)

print(f"✓ Download complete!")
print(f"Dataset location: {path}")