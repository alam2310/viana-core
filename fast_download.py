import os
import sys

# 1. FORCE ENABLE RUST ACCELERATOR (Must be done before importing huggingface_hub)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    import hf_transfer
    print("✅ hf_transfer (Rust accelerator) is installed and enabled.")
except ImportError:
    print("❌ Error: hf_transfer is NOT installed. Speed will be slow.")
    print("   Run: pip install hf_transfer")
    sys.exit(1)

from huggingface_hub import snapshot_download

# Configuration
REPO_ID = "iisc-aim/UVH-26"
DEST_DIR = "/root/Work/ViAna/data/raw/UVH-26"

print(f"🚀 Starting download of {REPO_ID}...")
print(f"📂 Destination: {DEST_DIR}")

# 2. Download with maximum concurrency
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=DEST_DIR,
    allow_patterns=["*.yaml", "*.txt", "*.jpg", "*.png", "*.json", "images/*", "labels/*"],
    resume_download=True,
    max_workers=64 
)

print("✅ Download Complete.")
