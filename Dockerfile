# ==============================================================================
# STAGE 1: The Builder (OpenCV Compilation) - NO CHANGES
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# 1. Install Build Dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git pkg-config python3-dev python3-numpy python3-pip \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    libgl1 libglib2.0-0 libprotobuf-dev protobuf-compiler \
    libjpeg-dev libpng-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Clone OpenCV (v4.10.0)
RUN git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv_contrib.git

# 3. Configure & Compile (CUDA ON, NVCUVID OFF)
RUN mkdir -p opencv/build && cd opencv/build && \
    cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D WITH_CUBLAS=ON \
    -D CUDA_ARCH_BIN=8.6 -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 \
    -D WITH_OPENGL=ON -D BUILD_opencv_python3=ON -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_PROTOBUF=OFF -D PROTOBUF_UPDATE_FILES=ON \
    -D BUILD_examples=OFF -D BUILD_tests=OFF -D BUILD_perf_tests=OFF \
    .. && \
    make -j$(nproc) && \
    make install

# ==============================================================================
# STAGE 2: The Runtime (Production Ready) - FIXED
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /workspace/ViAna
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Runtime Dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-numpy \
    libgl1 libglib2.0-0 libgomp1 libprotobuf-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
    libavcodec58 libavformat58 libswscale5 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install PyTorch (Pinned to CUDA 12.4)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install Data Engineering Tools (FIXED)
# - huggingface_hub[cli]: ADDS the 'huggingface-cli' command
# - hf_transfer: ADDS the Rust accelerator for speed
RUN pip3 install --no-cache-dir --default-timeout=100 \
    "huggingface_hub[cli]" \
    hf_transfer \
    tqdm \
    pytest \
    "numpy<2"\
    pandas \ 
    pyyaml \
    opencv-python \
    matplotlib \
    seaborn

# 4. Copy Compiled OpenCV Artifacts
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/include/opencv4 /usr/local/include/opencv4
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10

# 5. Setup Paths
RUN ldconfig
# [NEW] Fix for Absolute Symlinks in Dataset
# Creates a bridge so links pointing to '/root/Work/ViAna' 
# automatically redirect to the container mount '/app/ViAna'
RUN mkdir -p /root/Work && \
    ln -s /app/ViAna /root/Work/ViAna

ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages

CMD ["/bin/bash"]




