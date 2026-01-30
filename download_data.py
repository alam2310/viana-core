import os
from huggingface_hub import snapshot_download

# Configuration
REPO_ID = "iisc-aim/UVH-26"
DEST_DIR = "/root/Work/ViAna/data/raw/UVH-26"

print(f"🚀 Starting download from {REPO_ID}...")
print(f"📂 Destination: {DEST_DIR}")
print("⚡ resuming download with HIGH CONCURRENCY (64 threads)...")

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=DEST_DIR,
    # Patterns to get images, labels, and config
    allow_patterns=["*.yaml", "*.txt", "*.jpg", "*.png", "*.json", "images/*", "labels/*"],
    # Max workers increased to 64 to crush the "small file" bottleneck
    max_workers=64
)

print("✅ Download Complete.")
