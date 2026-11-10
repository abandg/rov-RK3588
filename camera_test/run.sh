#!/usr/bin/env bash
set -euo pipefail

test_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
build_dir="$test_dir/build"
device="${1:-/dev/video0}"

cmake -S "$test_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" -j2
exec "$build_dir/rk3588_vision" "$device"
