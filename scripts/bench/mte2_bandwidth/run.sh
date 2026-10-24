#!/bin/bash
# Run the MTE2 streaming benchmark inside a task-submit job.
#
# Usage: run.sh [blockDim]
#   blockDim must not exceed the board's AIC count. Default 28 (950PR 9579);
#   pass 32 on Ascend950DT_9582. TASK_DEVICE selects the device.
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
CANN_ROOT=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
if [ -f "$CANN_ROOT/set_env.sh" ]; then
    # shellcheck disable=SC1090
    source "$CANN_ROOT/set_env.sh"
fi

BD=${1:-28}
echo "MTE2 run: TASK_DEVICE=${TASK_DEVICE:-unset} blockDim=$BD"
exec "$HERE/build/mte2_bench" "$BD"
