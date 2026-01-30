import os
import sys

# 1. Force High-Speed Transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    import hf_transfer
    print("✅ Rust Accelerator (hf_transfer) is ACTIVE.")
except ImportError:
    print("❌ Warning: hf_transfer not found. Install it for speed.")

from huggingface_hub import snapshot_download

REPO_ID = "iisc-aim/UVH-26"
# Expand the ~ to the full path on your host
DEST_DIR = os.path.expanduser("~/Work/ViAna/data/raw/UVH-26")

print(f"🚀 Starting Host-Side Download: {REPO_ID}")
print(f"📂 Target: {DEST_DIR}")

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=DEST_DIR,
    # Download images, labels, and config
    allow_patterns=["*.yaml", "*.txt", "*.jpg", "*.png", "*.json", "images/*", "labels/*"],
    resume_download=True,
    max_workers=64 
)

print("✅ Download Complete. You may now check inside the container.")
