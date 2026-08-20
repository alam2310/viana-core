# Environment setup (from scratch)

Step-by-step guide to run **ViAna v2** on Ubuntu with Docker and NVIDIA GPUs. For day-to-day commands after setup, see [`DEPLOYMENT.md`](DEPLOYMENT.md).

**Target stack**

| Layer | Choice | Why |
|-------|--------|-----|
| Host OS | Ubuntu 24.04 LTS | Tested dev host |
| NVIDIA driver | 550+ (590 tested) | GPU passthrough to container |
| Container base | Ubuntu 22.04 + CUDA 12.4 | Stable OpenCV / PyTorch wheels |
| OpenCV | 4.10 compiled with CUDA | GPU image ops (see `Dockerfile`) |
| PyTorch | 2.6 + cu124 | Matches container CUDA |
| Video encode | FFmpeg `hevc_nvenc` | Small processed MP4 output |
| Engine | `python -m viana` | CLI + orchestrator workers |
| API | uvicorn `:8000` | `docker-compose.yml` |
| UI | Next.js on host | `apps/web` → `http://localhost:8000` |

Historical rationale and failed experiments: [`archive/ITVA_RESEARCH_LOG.md`](archive/ITVA_RESEARCH_LOG.md).

---

## Phase 1 — Host preparation

### 1.1 Verify NVIDIA driver

```bash
nvidia-smi
```

You should see your GPU(s) and driver version. If the command is missing:

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 1.2 Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker "$USER"   # log out and back in
```

### 1.3 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 1.4 Smoke test GPU in Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Phase 2 — Clone repository and data mount

```bash
git clone <your-remote> ViAna
cd ViAna
mkdir -p data/raw data/viana-outputs
```

`docker-compose.yml` mounts:

| Host | Container | Purpose |
|------|-----------|---------|
| `.` | `/app/ViAna` | Live code mount |
| `./data` (or `VIANA_DATA_ROOT`) | `/data` | Videos, outputs, optional training data |

Put review/test videos under `data/raw/`. Job outputs land in `data/viana-outputs/{project_id}/`.

---

## Phase 3 — Build and start the container

From repo root:

```bash
docker compose build
docker compose up -d
docker compose exec dev bash
```

Inside the container, working directory is `/app/ViAna`.

The compose service installs editable package deps on start, pins **NumPy &lt; 2**, and installs **trackers** (ByteTrack) without pulling NumPy 2.

---

## Phase 4 — Verify the v2 stack

Run **inside** the container:

```bash
cd /app/ViAna

# Editable install (if image predates latest pyproject)
pip install -e ".[dev]"
pip install -q "numpy>=1.26.0,<2"
pip install -q "trackers==2.6.0" --no-deps

# GPU + OpenCV CUDA
python3 -c "
import cv2, torch
print('OpenCV CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())
print('PyTorch GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'FAILED')
"

# FFmpeg NVENC (processed video)
ffmpeg -hide_banner -encoders 2>/dev/null | grep hevc_nvenc

# Engine CLI
python -m viana --help

# Orchestrator (already running via compose on :8000)
curl -s http://localhost:8000/health

# Unit tests
pytest tests/viana/ -q

# EasyOCR (first prescan downloads detection/recognition weights; corner ROI OCR)
python3 -m viana prescan --source /data/raw/hiv000001_inframe.mp4 --project-id smoke --frame-offset 0
```

Expected health response includes `"phase": 6`.

### Optional: process a short clip

```bash
export YOLO_CONFIG_DIR=/tmp/Ultralytics
python -m viana run --config /path/to/job.json
```

Job JSON must include `source_video_path`, `horizon_line`, `counting_line`, and `output_dir` under `/data/...`.

---

## Phase 5 — Next.js UI (host)

```bash
cd apps/web
cp .env.example .env.local
# NEXT_PUBLIC_API_URL=http://localhost:8000
# NEXT_PUBLIC_USE_MOCKS=false
npm install
npm run dev
```

Open the dashboard, run prescan, submit jobs against the live API.

---

## Phase 6 — Optional UVH-26 dataset (retrain only)

Only needed to **retrain** the vehicle model. See [`../../training/README.md`](../../training/README.md).

```bash
# On host or in container (with HF token)
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download visual-layer/uvh26 --repo-type dataset \
  --local-dir data/datasets/uvh26 --local-dir-use-symlinks False
```

Then follow `training/README.md` for convert → build manifest → train.

---

## Daily workflow

| Action | Command (host, repo root) |
|--------|----------------------------|
| Start | `docker compose up -d` |
| Shell | `docker compose exec dev bash` |
| API logs | `docker compose logs -f` |
| Stop | `docker compose down` |
| Rebuild after Dockerfile change | `docker compose build --no-cache` |

Long jobs: see [`TMUX_README.md`](TMUX_README.md).

---

## Known pitfalls

| Symptom | Cause | Fix |
|---------|--------|-----|
| OpenCV import error after extra pip install | NumPy 2.x pulled in | Image already pins `numpy>=1.26,<2`; reinstall that pin if you add packages |
| Tracker install upgrades NumPy | `trackers` deps | Image installs `trackers==2.6.0 --no-deps`; do not `pip install trackers` with deps |
| Compose start still pip-installs | Stale image predating Step 6.1 | `docker compose build` then `up` |
| No GPU in container | Toolkit not configured | Repeat Phase 1.3 |
| Huge processed MP4 | Wrong encoder | v2 uses HEVC NVENC cq 42 (see `src/viana/stages/render.py`) |
| Empty `_15min.csv` | No wall-clock on job | Set OCR / user start via prescan (corner ROI OCR in prescan) |
| First prescan slow or stuck at PRESCAN_RUNNING | EasyOCR weights missing after image rebuild; `english_g2` download from GitHub can stall | Image bakes CRAFT + `english_g2` at build time; cached under `/root/.EasyOCR/model` |
| Build fails on CUDA 13 / Ubuntu 24 base | Breaking toolchain | Stay on CUDA **12.4** + Ubuntu **22.04** in `Dockerfile` |
| OpenCV `NVCUVID` link errors | Deprecated in CUDA 12 | Dockerfile builds with NVCUVID **off**; use FFmpeg for video I/O |

---

## What the Dockerfile does (summary)

1. **Builder stage:** compile OpenCV 4.10 + contrib with CUDA, cuDNN, arch 8.6 (RTX 3060).  
2. **Runtime stage:** CUDA 12.4 runtime, FFmpeg, PyTorch cu124, Ultralytics, EasyOCR.  
3. **Symlink:** `/root/Work/ViAna` → `/app/ViAna` for older dataset paths.  
4. **Editable install:** `pip install -e ".[dev]"` for `viana` and `orchestrator`.  
5. **NumPy / ByteTrack:** after the editable install, re-pin `numpy>=1.26,<2` and `trackers==2.6.0 --no-deps` (Step 6.1). Compose starts uvicorn only.  
6. **EasyOCR weights:** English CRAFT + `english_g2` downloaded at build so first prescan does not hit GitHub.

Full build recipe: [`../../Dockerfile`](../../Dockerfile).
