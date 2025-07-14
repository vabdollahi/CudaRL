#!/bin/bash

#!/usr/bin/env bash
set -e

BUILD_DIR="build"
USE_CUDA="OFF"

# Parse arguments
for arg in "$@"; do
    if [[ "$arg" == "cuda" || "$arg" == "--cuda" ]]; then
        USE_CUDA="ON"
    fi
done

echo "🧹 Cleaning old build directory..."
rm -rf "$BUILD_DIR"

echo "📁 Creating new build directory..."
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

echo "⚙️ Configuring CMake (USE_CUDA=$USE_CUDA)..."
cmake .. -DUSE_CUDA=$USE_CUDA -DCMAKE_BUILD_TYPE=Release

echo "🔨 Building project..."
cmake --build . -- -j$(nproc)

echo "✅ Build complete!"
