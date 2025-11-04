#!/bin/bash
# Build script for Blake2b GPU Miner

set -e

# Detect CUDA installation
if [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
elif [ -d "/opt/cuda" ]; then
    CUDA_PATH="/opt/cuda"
elif [ -d "/Developer/NVIDIA/CUDA" ]; then
    CUDA_PATH=$(ls -d /Developer/NVIDIA/CUDA-* | tail -1)
else
    echo "Error: CUDA not found. Please install CUDA toolkit."
    echo "Expected locations: /usr/local/cuda, /opt/cuda, or /Developer/NVIDIA/CUDA-*"
    exit 1
fi

echo "Using CUDA at: $CUDA_PATH"

# Check for nvcc
if [ ! -f "$CUDA_PATH/bin/nvcc" ]; then
    echo "Error: nvcc not found at $CUDA_PATH/bin/nvcc"
    exit 1
fi

# Compile CUDA files
echo "Compiling CUDA files..."
mkdir -p build/obj

# Compile kernels.cu
echo "  Compiling kernels.cu..."
$CUDA_PATH/bin/nvcc \
    -gencode=arch=compute_50,code=sm_50 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_86,code=sm_86 \
    -gencode=arch=compute_89,code=sm_89 \
    -O3 -use_fast_math -Xcompiler -fPIC -c \
    kernels.cu -o build/obj/kernels.o

# Compile blake2b.cu
echo "  Compiling blake2b.cu..."
$CUDA_PATH/bin/nvcc \
    -gencode=arch=compute_50,code=sm_50 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_86,code=sm_86 \
    -gencode=arch=compute_89,code=sm_89 \
    -O3 -use_fast_math -Xcompiler -fPIC -c \
    blake2b.cu -o build/obj/blake2b.o

echo "CUDA files compiled successfully!"
echo ""
echo "Now run: npm install"
echo "Or: node-gyp rebuild"
