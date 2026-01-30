# 🛠️ ITVA Environment Setup Guide: "Golden Master" v3.0

**Version:** 3.0 (Finalized)
**Target Audience:** Engineering Team (Entry Level +)
**Status:** ✅ Production Ready
**Objective:** Set up a GPU-accelerated Computer Vision environment (ViAna) from scratch.

## **1. The Target Stack Strategy**
To achieve high-throughput Video Analytics (ViAna), we architected a custom stack that minimizes CPU bottlenecks and maximizes GPU throughput.

* **Host Layer:** Ubuntu 24.04 + NVIDIA Driver 590 (Bleeding edge host)
* **Container Base:** Ubuntu 22.04 LTS (Stable production base)
* **Compute Core:** CUDA 12.4 + cuDNN 9 (Pinned for stability)
* **Vision Engine:** OpenCV 4.10.0 (Custom compiled for CUDA)
* **AI Framework:** PyTorch 2.6 (Pinned explicitly to CUDA 12.4)
* **Data Layer:** Symlink-bridged UVH-26 Dataset

---

## **2. Construction Steps (The "Recipe")**
These are the specific engineering steps taken to build the environment from scratch.

### **Step A: The GPU Foundation (CUDA & Drivers)**
We deliberately avoided standard package managers (`apt install python3-opencv`) because they lack GPU support.
1.  **Base Image:** We utilized `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` to gain access to the full NVIDIA compiler toolchain (`nvcc`) required for compiling custom libraries.
2.  **Driver Compatibility:** We verified the Host Driver (v590) backward compatibility with the Container's CUDA Toolkit (v12.4) to ensure seamless GPU passthrough.

### **Step B: Custom OpenCV Compilation (The Heavy Lift)**
Standard OpenCV does not support NVIDIA GPUs out of the box. To fix this, we implemented a **Multi-Stage Docker Build** to compile OpenCV from source:
* **Source:** Cloned OpenCV & OpenCV Contrib v4.10.0.
* **Key Build Flags Enabled:**
    * `WITH_CUDA=ON`, `WITH_CUDNN=ON`, `OPENCV_DNN_CUDA=ON`: Activates the GPU backend.
    * `CUDA_ARCH_BIN=8.6`: Optimizes binary specifically for Ampere architecture (RTX 3060).
    * `ENABLE_FAST_MATH=1`: Trades negligible precision for significant speed gains.
    * `WITH_CUBLAS=ON`: Leverages NVIDIA's optimized linear algebra libraries.
* **Outcome:** A custom `cv2.so` Python binding that offloads image processing to the GPU.

### **Step C: PyTorch Pinning**
To prevent the common "PyTorch vs. System CUDA" version mismatch:
* **Strategy:** We bypassed the default PyPi index.
* **Action:** Installed PyTorch pointing explicitly to the NVIDIA CUDA 12.4 wheel index (`https://download.pytorch.org/whl/cu124`).
* **Result:** PyTorch runtime exactly matches the container's CUDA drivers, eliminating "driver mismatch" errors.

---------------------------------------------------------------------------------------------------
### **STEPS BEGIN HERE**
---------------------------------------------------------------------------------------------------

## **Phase 1: Host Machine Preparation**
*Before touching the project code, we must ensure the physical machine is ready to talk to the GPU.*

### **Step 1.1: Verify NVIDIA Drivers**
Ensure your host machine (Ubuntu 24.04) has the NVIDIA drivers installed and running.
```bash
# Run this command in your terminal
nvidia-smi
Success: You see a table listing your GPU (e.g., RTX 3060) and Driver Version (e.g., 590.x).

Failure: If command not found, install drivers via sudo ubuntu-drivers autoinstall and reboot.

Step 1.2: Install Docker & NVIDIA Container Toolkit
Docker needs a special toolkit to access your GPU. Standard Docker installation is not enough.

Bash
# 1. Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin

# 2. Add NVIDIA Container Toolkit Repository
curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. Install the Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 4. Configure Docker to use GPU
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
Phase 2: Project Initialization
Now we set up the codebase and directory structure.

Step 2.1: Clone/Create Repository
Set up the standard directory structure on your local machine.

Bash
# Create the project root
mkdir -p ViAna/data/dataset
mkdir -p ViAna/src/utils
mkdir -p ViAna/configs
mkdir -p ViAna/docs

cd ViAna
Step 2.2: Add Configuration Files
Ensure the following files are present in the root directory (refer to the repository for file contents):

Dockerfile (The build recipe)

docker-compose.yml (The runtime manager)

requirements.txt (If used locally, though we rely on Docker)

Phase 3: Dataset Acquisition (Hugging Face)
We use Hugging Face (HF) to manage our datasets. Since our dataset is private or gated, you need authentication.

Step 3.1: Install HF CLI (Locally)
You need the command-line tool to download data efficiently.

Bash
pip install -U "huggingface_hub[cli]"
Step 3.2: Authenticate
You will need your User Access Token from Hugging Face Settings.

Bash
huggingface-cli login
# Paste your token when prompted (it will not show on screen).
Step 3.3: Download the Dataset
We download the data specifically to the data/ folder so it can be mounted into Docker later.

Bash
# Navigate to the data storage folder
cd data/

# Download the UVH-26 dataset (Replace 'Org/Repo' with actual ID if different)
# --repo-type dataset: Specifies we are downloading data, not models
# --local-dir .: Downloads contents directly to current folder
huggingface-cli download visual-layer/uvh26 --repo-type dataset --local-dir ./dataset --local-dir-use-symlinks False

# Return to root
cd ..
Note: We use --local-dir-use-symlinks False to ensure real files are downloaded, preventing permission issues.

Phase 4: Docker Environment Setup
This is where we compile the custom environment with CUDA support.

Step 4.1: Build the "Golden Image"
This step compiles OpenCV from source. It will take ~15-20 minutes. Do not interrupt it.

Bash
# Make sure you are in the folder containing 'Dockerfile'
docker compose build
Step 4.2: Launch the Environment
Start the container in the background ("Detached" mode).

Bash
docker compose up -d
Check status: Run docker ps. You should see a container named viana_core running.

Step 4.3: Enter the Development Console
Log in to your active development environment.

Bash
docker compose exec dev bash
You are now inside the container. Your terminal prompt should change to root@viana_core:/workspace/ViAna#.

Phase 5: Final Verification
Run these commands INSIDE the container to verify the setup is successful.

Step 5.1: Verify Directory Mounts
Check if the dataset you downloaded in Phase 3 is visible here.

Bash
ls -F data/dataset/
Success: You should see data.yaml, images/, labels/.

Step 5.2: The "Smoke Test"
Verify GPU acceleration for AI (PyTorch) and Vision (OpenCV).

Bash
python3 -c '
import cv2, torch, sys
print(f"✅ Python Version: {sys.version.split()[0]}")
print(f"✅ OpenCV GPU Devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
print(f"✅ PyTorch GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "FAILED"}")
'
Step 5.3: Verify Classification Logic
Run the included classifier test to ensure code logic is working.

Bash
pytest tests/
Cheatsheet: Daily Workflow
Action	Command (Run from Host)
Start Work	docker compose up -d
Enter Container	docker compose exec dev bash
Stop Work	docker compose down
Rebuild (after Dockerfile change)	docker compose build
View Logs	docker compose logs -f

## Manifest Oversampling**. Multiplied rare classes (Mini Bus 20x, LCV 5x) in the training manifest.

# 1. Update the vehicle tasonomy JSON (no need to run as updated mapping is comitted)
python3 src/utils/sync_taxonomy.py

# 2. Phase 1.3: Manifest Oversampling
python3 src/utils/build_itva_dataset.py

Phase 1.4: Training dataset:
python3 src/train.py
