import os
import socket
import urllib3.util.connection as urllib3_cn
from huggingface_hub import snapshot_download

# --- PATCH: FORCE IPV4 (Fixes Docker/IPv6 Timeout) ---
def allowed_gai_family():
    return socket.AF_INET

# Apply the patch to urllib3 (used by requests & huggingface_hub)
urllib3_cn.allowed_gai_family = allowed_gai_family
print("🛡️  Network Patch: Forced IPv4 (Bypassing IPv6 timeouts)")

# --- CONFIGURATION ---
REPO_ID = "iisc-aim/UVH-26"
DEST_DIR = "/root/Work/ViAna/data/raw/UVH-26"

# Force Rust Accelerator
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print(f"🚀 Starting Turbo Download: {REPO_ID}")
print(f"📂 Destination: {DEST_DIR}")

# --- EXECUTION ---
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=DEST_DIR,
    allow_patterns=["*.yaml", "*.txt", "*.jpg", "*.png", "*.json", "images/*", "labels/*"],
    resume_download=True,
    max_workers=64
)

print("✅ Download Complete.")
