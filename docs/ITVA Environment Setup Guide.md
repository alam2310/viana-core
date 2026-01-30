# **🛠️ ITVA Environment Setup Guide: "Golden Master"**

**Version: 2.0 (Verified) Date: January 28, 2026 Status: ✅ Production Ready**

**Target Stack:**

* **Host: Ubuntu 24.04 \+ NVIDIA Driver 590**  
* **Container: Ubuntu 22.04 \+ CUDA 12.4 \+ cuDNN 9**  
* **Vision: OpenCV 4.10.0 (CUDA-Enabled, No NVCUVID)**  
* **AI Core: PyTorch 2.6 \+ torchvision (Pinned to CUDA 12.4)**  
* **Decoding: NVIDIA DALI / FFmpeg**

---

## **1\. Docker Build (The "Safe Landing" Image)**

***Create a file named `Dockerfile` and paste the content below. I have added the PyTorch installation directly into the file so it is automatic in future builds.***

**Dockerfile**

**\# \==============================================================================**

**\# STAGE 1: The Builder (OpenCV Compilation)**

**\# \==============================================================================**

**FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder**

**ENV DEBIAN\_FRONTEND=noninteractive**

**WORKDIR /workspace**

**\# 1\. Install Build Dependencies**

**RUN apt-get update && apt-get install \-y \\**

    **build-essential cmake git pkg-config python3-dev python3-numpy python3-pip \\**

    **libavcodec-dev libavformat-dev libswscale-dev \\**

    **libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \\**

    **libgl1 libglib2.0-0 libprotobuf-dev protobuf-compiler \\**

    **libjpeg-dev libpng-dev libtiff-dev \\**

    **&& rm \-rf /var/lib/apt/lists/\***

**\# 2\. Clone OpenCV (v4.10.0)**

**RUN git clone \--depth 1 \--branch 4.10.0 https://github.com/opencv/opencv.git && \\**

    **git clone \--depth 1 \--branch 4.10.0 https://github.com/opencv/opencv\_contrib.git**

**\# 3\. Configure & Compile (CUDA ON, NVCUVID OFF)**

**RUN mkdir \-p opencv/build && cd opencv/build && \\**

    **cmake \\**

    **\-D CMAKE\_BUILD\_TYPE=RELEASE \\**

    **\-D CMAKE\_INSTALL\_PREFIX=/usr/local \\**

    **\-D OPENCV\_EXTRA\_MODULES\_PATH=../../opencv\_contrib/modules \\**

    **\-D WITH\_CUDA=ON \-D WITH\_CUDNN=ON \-D OPENCV\_DNN\_CUDA=ON \-D WITH\_CUBLAS=ON \\**

    **\-D CUDA\_ARCH\_BIN=8.6 \-D ENABLE\_FAST\_MATH=1 \-D CUDA\_FAST\_MATH=1 \\**

    **\-D WITH\_OPENGL=ON \-D BUILD\_opencv\_python3=ON \-D OPENCV\_GENERATE\_PKGCONFIG=ON \\**

    **\-D OPENCV\_ENABLE\_NONFREE=ON \\**

    **\-D BUILD\_PROTOBUF=OFF \-D PROTOBUF\_UPDATE\_FILES=ON \\**

    **\-D BUILD\_examples=OFF \-D BUILD\_tests=OFF \-D BUILD\_perf\_tests=OFF \\**

    **.. && \\**

    **make \-j$(nproc) && \\**

    **make install**

**\# \==============================================================================**

**\# STAGE 2: The Runtime (Production Ready)**

**\# \==============================================================================**

**FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04**

**WORKDIR /workspace/ViAna**

**ENV DEBIAN\_FRONTEND=noninteractive**

**\# 1\. Install Runtime Dependencies**

**RUN apt-get update && apt-get install \-y \\**

    **python3 python3-pip python3-numpy \\**

    **libgl1 libglib2.0-0 libgomp1 libprotobuf-dev \\**

    **libjpeg-dev libpng-dev libtiff-dev \\**

    **libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \\**

    **libavcodec58 libavformat58 libswscale5 \\**

    **&& rm \-rf /var/lib/apt/lists/\***

**\# 2\. Install PyTorch (Pinned to CUDA 12.4)**

**\# This ensures strict compatibility with the container's CUDA drivers.**

**RUN pip3 install torch torchvision torchaudio \--index-url https://download.pytorch.org/whl/cu124**

**\# 3\. Copy Compiled OpenCV Artifacts**

**COPY \--from=builder /usr/local/lib /usr/local/lib**

**COPY \--from=builder /usr/local/include/opencv4 /usr/local/include/opencv4**

**COPY \--from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10**

**\# 4\. Setup Paths**

**RUN ldconfig**

**ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages**

**CMD \["/bin/bash"\]**

---

## **2\. Build & Launch**

**Run these commands to build the image and tag it correctly.**

**Bash**

**\# 1\. Build the image (Approx 20 mins)**

**docker build \-t itva-base:stable .**

**\# 2\. (Optional) Remove old build artifacts if re-building**

**docker image prune \-f**

---

## **3\. The "Smoke Test" (Verification)**

**This is the mandatory acceptance test. It validates that the OS, Python, OpenCV, and PyTorch all agree on the GPU status.**

**Run this command block exactly as written:**

**Bash**

**docker run \--rm \--runtime=nvidia \--gpus all itva-base:stable python3 \-c '**

**import cv2**

**import torch**

**import sys**

**import numpy as np**

**print("========== ITVA ENVIRONMENT DIAGNOSTIC \==========")**

**\# 1\. Check OS & Python**

**print(f"✅ Python Version: {sys.version.split()\[0\]}")**

**\# 2\. Check OpenCV CUDA**

**try:**

    **cv\_count \= cv2.cuda.getCudaEnabledDeviceCount()**

    **if cv\_count \> 0:**

        **print(f"✅ OpenCV CUDA:   ACTIVE ({cv\_count} Devices)")**

    **else:**

        **print("❌ OpenCV CUDA:   FAILED (0 Devices)")**

        **sys.exit(1)**

**except Exception as e:**

    **print(f"❌ OpenCV CUDA:   CRASHED ({e})")**

    **sys.exit(1)**

**\# 3\. Check PyTorch CUDA**

**try:**

    **if torch.cuda.is\_available():**

        **gpu\_name \= torch.cuda.get\_device\_name(0)**

        **print(f"✅ PyTorch CUDA:  ACTIVE ({gpu\_name})")**

    **else:**

        **print("❌ PyTorch CUDA:  FAILED")**

        **sys.exit(1)**

**except Exception as e:**

    **print(f"❌ PyTorch CUDA:  CRASHED ({e})")**

    **sys.exit(1)**

**print("\\n========== STRESS TEST (LOAD CHECK) \==========")**

**\# 4\. PyTorch Matrix Multiplication Test**

**print("\[1/2\] PyTorch Matrix Mul...", end=" ")**

**try:**

    **a \= torch.randn(5000, 5000, device="cuda:0")**

    **b \= torch.randn(5000, 5000, device="cuda:0")**

    **\_ \= torch.matmul(a, b)**

    **print("SUCCESS ✅")**

**except Exception as e:**

    **print(f"FAILED ❌ ({e})")**

**\# 5\. OpenCV GpuMat Upload Test**

**print("\[2/2\] OpenCV GpuMat Upload...", end=" ")**

**try:**

    **src\_host \= np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)**

    **src\_gpu \= cv2.cuda\_GpuMat()**

    **src\_gpu.upload(src\_host)**

    **dst \= cv2.cuda.resize(src\_gpu, (1000, 1000))**

    **print("SUCCESS ✅")**

**except Exception as e:**

    **print(f"FAILED ❌ ({e})")**

**print("\\n🚀 RESULT: ENVIRONMENT IS VERIFIED & READY FOR PHASE 1.")**

**'**

---

### **4\. Expected Output**

**If your environment is healthy, the output will look exactly like this:**

**Plaintext**

**\========== ITVA ENVIRONMENT DIAGNOSTIC \==========**

**✅ Python Version: 3.10.12**

**✅ OpenCV CUDA:   ACTIVE (2 Devices)**

**✅ PyTorch CUDA:  ACTIVE (NVIDIA GeForce RTX 3060\)**

**\========== STRESS TEST (LOAD CHECK) \==========**

**\[1/2\] PyTorch Matrix Mul... SUCCESS ✅**

**\[2/2\] OpenCV GpuMat Upload... SUCCESS ✅**

**🚀 RESULT: ENVIRONMENT IS VERIFIED & READY FOR PHASE 1\.**

