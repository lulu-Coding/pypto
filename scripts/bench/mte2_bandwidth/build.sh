#!/bin/bash
# Build the pure GM->L1 MTE2 streaming benchmark.
#
# Environment:
#   ASCEND_HOME_PATH          CANN root; sourced from set_env.sh when present.
#   SOC_VERSION               SoC for the kernel build (default ascend950pr_9579).
#   ASCEND_CANN_PACKAGE_PATH  Overrides the CANN root used by CMake.
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
CANN_ROOT=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
if [ -f "$CANN_ROOT/set_env.sh" ]; then
    # shellcheck disable=SC1090
    source "$CANN_ROOT/set_env.sh"
    CANN_ROOT=${ASCEND_HOME_PATH:-$CANN_ROOT}
fi
SOC=${SOC_VERSION:-ascend950pr_9579}

cd "$HERE"
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
    -DSOC_VERSION="$SOC" > build_cfg.log 2>&1 || { echo CFG_FAIL; tail -30 build_cfg.log; exit 1; }
cmake --build build --parallel "${PYPTO_BUILD_JOBS:-2}" > build.log 2>&1 || {
    echo BUILD_FAIL
    tail -60 build.log
    exit 1
}
echo "BUILD_OK soc=$SOC cann=$CANN_ROOT"
