🛠️ ITVA Environment Setup Guide: "Golden Master" v3.0
Version: 3.0 (Finalized)

Status: ✅ Production Ready

Scope: Docker Container, Dependencies, Project Structure, and Workflow.

1. Project Directory Structure
Before building, ensure your local project follows this standardized structure. This separates logic (src), configuration (configs), and data (data).

Plaintext
ViAna/
├── .gitignore              <-- Ignores data/ and __pycache__/
├── README.md
├── docker-compose.yml      <-- Workflow Engine (Refer to repo)
├── Dockerfile              <-- Build Definition (Refer to repo)
├── main.py                 <-- Entry Point
├── audit_dataset.py        <-- Dataset Auditor Tool
├── process_video.py        <-- Video Processing Pipeline
│
├── configs/
│   └── vehicle_taxonomy.json   <-- Classification Logic
│
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── classifier.py       <-- VehicleClassifier Class
│   │   └── dataset_auditor.py  <-- DatasetAuditor Class
│   └── core/
│       └── __init__.py
│
├── tests/
│   ├── __init__.py
│   └── test_classifier.py      <-- Unit Tests
│
├── data/                   <-- MOUNTED VOLUME (Not in Git)
│   ├── dataset/            <-- Contains data.yaml
│   ├── raw_videos/
│   └── outputs/
│
└── docs/
    └── VEHICLE_CLASSIFICATION.md
2. Build & Launch Procedure
We use Docker Compose to manage the environment. This ensures the GPU is attached, shared memory is sufficient, and the volume is mounted correctly.

Step 1: Build the Image
Run this command from the project root. It reads the Dockerfile and docker-compose.yml.

Bash
docker compose build
Step 2: Start the Environment
Run this to start the container in the background (detached mode).

Bash
docker compose up -d
Step 3: Enter the Container
This drops you into the shell inside the container to run scripts.

Bash
docker compose exec dev bash
Step 4: Stop the Environment
When finished, shut down the container cleanly.

Bash
docker compose down
3. Verification (The Smoke Test)
Once inside the container (docker compose exec dev bash), run this one-liner to verify the OS, Python, OpenCV (CUDA), and PyTorch are communicating correctly.

Bash
python3 -c '
import cv2, torch, sys
print(f"✅ Python: {sys.version.split()[0]}")
print(f"✅ OpenCV CUDA: {cv2.cuda.getCudaEnabledDeviceCount()} Devices")
print(f"✅ PyTorch CUDA: {torch.cuda.get_device_name(0)}")
try:
    a = torch.randn(5000, 5000, device="cuda:0")
    b = torch.randn(5000, 5000, device="cuda:0")
    torch.matmul(a, b)
    print("✅ Matrix Mul: SUCCESS")
except: print("❌ Matrix Mul: FAILED")
'
Expected Output:

Plaintext
✅ Python: 3.10.12
✅ OpenCV CUDA: 1 Devices
✅ PyTorch CUDA: NVIDIA GeForce RTX 3060
✅ Matrix Mul: SUCCESS


Action,             Command (Run from Host)
Start Work,         docker compose up -d
Enter Container,    docker compose exec dev bash
Stop Work,          docker compose down
Rebuild (change),   docker compose build
View Logs,          docker compose logs -f

---- using SSH: ----

Kill the GUI: 
> sudo systemctl isolate multi-user.target

How to get the GUI back
> sudo systemctl isolate graphical.target