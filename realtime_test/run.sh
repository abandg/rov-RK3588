#!/usr/bin/env bash
set -euo pipefail

test_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
runtime_dir="$test_dir/runtime"
build_dir="$test_dir/build-v16"
device="${1:-/dev/video0}"
model="${2:-"$test_dir/../yolov8n_rk3588.rknn"}"

cmake -S "$test_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DRKNN_RUNTIME_LIBRARY="$runtime_dir/librknnrt.so"
cmake --build "$build_dir" -j2
LD_LIBRARY_PATH="$runtime_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    exec "$build_dir/realtime_test" "$model" "$device"
