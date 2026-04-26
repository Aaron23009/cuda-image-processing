#!/usr/bin/env bash
set -e
echo "Building CUDA Image Processing Pipeline..."
make clean build
echo "Running grayscale conversion on input images..."
make run
echo ""
echo "Running with Gaussian blur..."
make run-blur
echo ""
echo "Done! Check data/results.csv for timing analysis."
