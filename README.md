# CUDA GPU Image Processing Pipeline
## GPU Specialization Capstone Project

A high-performance GPU-accelerated batch image processing pipeline
implementing RGB to Grayscale conversion and Gaussian blur using
custom CUDA kernels. Demonstrates GPU vs CPU speedup across 100+
images with detailed timing analysis.

## Features
- RGB to Grayscale conversion using ITU-R BT.709 luminance formula
- Optional 3x3 Gaussian blur filter
- GPU vs CPU timing comparison with speedup metrics
- Batch processing of 100+ images
- CSV logging of per-image results
- Detailed GPU device information output

## Algorithm
Two CUDA kernels implemented:

1. convertRGBToGrayscale: Maps a 2D thread grid to pixel coordinates.
   Each thread applies: Gray = 0.2126*R + 0.7152*G + 0.0722*B

2. applyGaussianBlur: Applies a 3x3 weighted Gaussian kernel to
   smooth the grayscale image, reducing noise.

## Requirements
- NVIDIA GPU (Compute Capability 3.0+)
- CUDA Toolkit 10.0+
- GCC/G++ with C++17 support
- stb_image (included in src/)

## Installation
    git clone https://github.com/Aaron23009/cuda-image-processing.git
    cd cuda-image-processing

## Usage
    # Build
    make build

    # Run grayscale conversion
    make run

    # Run with Gaussian blur
    make run-blur

    # Custom paths
    ./bin/image_processor.exe -i data/input -o data/output -l results.csv -b

## CLI Arguments
- -i   Input directory (default: data/input)
- -o   Output directory (default: data/output)
- -l   CSV log file path (default: data/results.csv)
- -b   Enable Gaussian blur (optional flag)

## Output
- Processed grayscale PNG images in data/output/
- CSV file with: filename, rows, columns, pixels, gpu_ms, cpu_ms, speedup

## Project Structure
    cuda-image-processing/
    |-- bin/                    Compiled executables
    |-- data/
    |   |-- input/              Input RGB images + manifest.txt
    |   |-- output/             Processed grayscale images
    |   +-- results.csv         GPU vs CPU timing results
    |-- src/
    |   |-- main.cu             CUDA source (kernels + host code)
    |   |-- stb_image.h         Header-only image loader
    |   +-- stb_image_write.h   Header-only image writer
    |-- Makefile
    |-- run.sh
    |-- README.md
    +-- INSTALL

## Results
On NVIDIA T4 GPU processing 100 images:
- Average GPU time: ~0.03 ms per image
- Average CPU time: ~2.5 ms per image
- Average speedup: ~80x faster on GPU
