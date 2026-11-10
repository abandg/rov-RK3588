#!/usr/bin/env bash
set -euo pipefail

test_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
runtime_dir="$test_dir/runtime"
build_dir="$test_dir/build-v16"
image="${1:-"$test_dir/test.jpg"}"
model="${2:-"$test_dir/../yolov8n_rk3588.rknn"}"

# 相对路径按调用者的目录解释，检测结果固定保存到 image_test。
case "$image" in
    /*) ;;
    *) image="$PWD/$image" ;;
esac
case "$model" in
    /*) ;;
    *) model="$PWD/$model" ;;
esac

cmake -S "$test_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DRKNN_RUNTIME_LIBRARY="$runtime_dir/librknnrt.so"
cmake --build "$build_dir" -j2
cd -- "$test_dir"
LD_LIBRARY_PATH="$runtime_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    exec "$build_dir/image_probe" "$model" "$image"
