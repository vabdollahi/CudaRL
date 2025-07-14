#!/bin/bash
mkdir -p build && cd build && cmake .. -DUSE_CUDA=ON && make -j$(nproc)