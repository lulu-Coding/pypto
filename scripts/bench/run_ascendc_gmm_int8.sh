#!/bin/bash
# Build and run the CANN aclnnGroupedMatmulWeightNz INT8 reference. The
# defaults match the PyPTO 10k kernel: 8192x3584x3072, G=32, per-group NZ.
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
INC="${ASCEND_HOME_PATH}/include"
LIB="${ASCEND_HOME_PATH}/lib64"
HERE="$(cd "$(dirname "$0")" && pwd)"
g++ -std=c++17 -O2 -I"$INC" "$HERE/ascendc_gmm_int8_bench.cpp" \
    -L"$LIB" -Wl,-rpath,"$LIB" -lopapi -lascendcl -lnnopbase -lpthread -ldl \
    -o "$HERE/ascendc_gmm_int8_bench"
export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH}"
export GMM_G="${GMM_G:-32}"
export GMM_M="${GMM_M:-8192}"
export GMM_K="${GMM_K:-3584}"
export GMM_N="${GMM_N:-3072}"
export GMM_REPS="${GMM_REPS:-5}"
echo "TASK_DEVICE=${TASK_DEVICE:-unset} G=${GMM_G} M=${GMM_M} K=${GMM_K} N=${GMM_N} reps=${GMM_REPS}"
exec "$HERE/ascendc_gmm_int8_bench"
