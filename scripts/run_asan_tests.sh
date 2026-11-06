#!/bin/bash
set -e
source /root/tinytensor_venv/bin/activate
cd /mnt/d/MERN/TinyTensor

export PYTHONPATH=/mnt/d/MERN/TinyTensor/python
export LD_PRELOAD="/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so /usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export ASAN_OPTIONS="detect_leaks=0"

pytest -v tests/python/
