# Shared A5 Test Environment

This appendix records a working PyPTO setup on a shared Ascend A5 test machine
behind a job-submission wrapper. It is a reference for reproducing the setup and
for reading numbers produced on that machine — not a supported install path for
every environment.

Angle-bracket values are site-specific: `<a5-host>`, `<ssh-port>`, `<user>`, and
`<a5-root>` (the working root on the shared machine, used as `/home/pypto-tools`
and `/home/pyptouser/luxiaolu` are below).

## Target

| Item | Value |
| ---- | ----- |
| Device | Ascend950PR 9579 (A5), board A310-50-C00MM304A1; `npu-smi info -t board` |
| Observed AI cores | 28 AIC at 1.65 GHz on all eight whitelisted devices; `npu-smi info -t common` |
| OS / compiler | Ubuntu 24.04, system GCC 13.3.0 |
| CANN | 9.3.0, `/usr/local/Ascend/ascend-toolkit` |
| NPU access | shared machine; NPU work goes through `task-submit`, never a bare `python` |

The PyPTO toolchain ships GCC 15.2.0 as `gcc-15` / `g++-15`, but its `bin`
directory has no unversioned `g++`, so CMake keeps using the system
`/usr/bin/g++` (13.3.0). That is the compiler the validated build used.

## Layout

```text
<a5-root>/
├── pypto/                       # superproject, branch group-matmul-i8
│   ├── runtime/                 # runtime, branch env/a5-cann93-buildfix
│   ├── 3rdparty/libbacktrace/
│   ├── 3rdparty/msgpack-c/
│   └── ...
├── venvs/pypto/                 # virtualenv
├── build_runtime.sh
├── build_pypto.sh
├── ptoas-0.60-cp310-cp310-manylinux_2_34_x86_64.whl
├── pypto-prefetch-extra/        # prefetch patches and untracked files
└── smoke/
    ├── pypto_smoke.py
    ├── run_smoke.sh
    ├── run_gmm_10g_profile.sh
    └── compile_10g.py
```

## Packages

| Component | Version / pin | How it was obtained |
| --------- | ------------- | ------------------- |
| PyPTO toolchain | 2026.08.5 | preinstalled under `/opt/pypto/toolchain`; activate with `pypto-toolchain export 2026.08.5` |
| Python | 3.10.18 | toolchain |
| uv | 0.9.28 | toolchain |
| ccache | 4.10.2 | toolchain |
| CMake / Ninja | system | preinstalled |
| CANN | 9.3.0 | preinstalled at `/usr/local/Ascend/ascend-toolkit` |
| PTOAS | v0.60 wheel | GitHub release asset, downloaded through `gh-proxy.com`; SHA256 `3a3f428a12b4f0a1bc61f978fd675417da543bab09494344426bb35977de842a` |
| PyPTO | branch `group-matmul-i8` @ `2eb7d3f8` | git bundle |
| runtime (`simpler`) | branch `env/a5-cann93-buildfix` @ `7186788f` | git bundle |
| pto-isa | `5a4f74cb` | git bundle |
| libbacktrace | `6f8310e` | git bundle |
| msgpack-c | `91990874` | git bundle |
| torch | 2.6.0+cpu | shared tree `/home/pypto-tools/python-packages/torch-cpu-2.6.0`, activated by `activate.sh` |
| numpy | 2.2.6 | uv + Aliyun PyPI mirror |
| pytest | 9.1.1 | uv + Aliyun PyPI mirror |
| cloudpickle | 3.1.2 | uv + Aliyun PyPI mirror |
| PyYAML | 6.0.3 | uv + Aliyun PyPI mirror |

`ptoas` on PyPI only has 0.1.x; the pinned release must be taken from the
project's GitHub releases. On this machine CANN's `set_env.sh` prepends its own
`bin` and `python/site-packages`, so `ptoas` resolves to the CANN-bundled binary,
which reports `ptoas 0.67`. The pip wheel is still staged for environments where
CANN does not override it. Verify the binary actually in use with
`command -v ptoas && ptoas --version`; do not infer it from the wheel tag.

## Shell setup

Every build or run starts from the same environment:

```bash
eval "$(pypto-toolchain export 2026.08.5)"
source <a5-root>/venvs/pypto/bin/activate
source /home/pypto-tools/python-packages/torch-cpu-2.6.0/activate.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Set `UV_INDEX_URL` to a reachable mirror before installing anything. On this
machine the Aliyun mirror was the one that worked:

```bash
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export UV_CONSTRAINT=<a5-root>/pypto/build-constraints.txt
```

## 1. Transfer the source tree

The shared machine reaches the fork's bundles, not necessarily GitHub. Build the
bundles on the workstation and copy them over SSH:

```bash
git bundle create pypto.bundle group-matmul-i8
git bundle create runtime.bundle feat/a5-sdma-prefetch
git bundle create libbacktrace.bundle <pin>
git bundle create msgpack-c.bundle <pin>
git bundle create pto-isa.bundle <pin>

scp -P <ssh-port> *.bundle <user>@<a5-host>:<a5-root>/pypto-transfer/bundles/
```

On the shared machine:

```bash
git clone -b group-matmul-i8 <a5-root>/pypto-transfer/bundles/pypto.bundle <a5-root>/pypto
cd <a5-root>/pypto
git clone -b env/a5-cann93-buildfix <a5-root>/pypto-transfer/bundles/runtime.bundle runtime
git clone <a5-root>/pypto-transfer/bundles/libbacktrace.bundle 3rdparty/libbacktrace
git clone <a5-root>/pypto-transfer/bundles/msgpack-c.bundle 3rdparty/msgpack-c
```

The bundles were also used as the `origin` of the shared clone. To update after a
new commit, regenerate `pypto.bundle`, copy it over the old file, then:

```bash
cd <a5-root>/pypto
git fetch origin group-matmul-i8
git pull --ff-only origin group-matmul-i8
```

The submodules are populated manually for the same connectivity reason.
`git submodule status` therefore shows a leading `-`; that is expected. Do **not**
run `git submodule update` — it tries to reach the upstream URLs and fails.

## 2. Create the virtualenv and install Python packages

```bash
/opt/pypto/toolchain/2026.08.5/python/bin/python3.10 -m venv <a5-root>/venvs/pypto
source <a5-root>/venvs/pypto/bin/activate

uv pip install --python <a5-root>/venvs/pypto/bin/python \
    numpy pytest cloudpickle pyyaml
```

Use the shared CPU torch tree instead of installing torch into the venv; the
default wheel pulls in roughly 2 GB of CUDA payload that this workflow never
uses. The activation script sets `PYTHONPATH` for the shared tree:

```bash
source /home/pypto-tools/python-packages/torch-cpu-2.6.0/activate.sh
```

## 3. Install PTOAS

```bash
uv pip install --no-deps --python <a5-root>/venvs/pypto/bin/python \
    <a5-root>/ptoas-0.60-cp310-cp310-manylinux_2_34_x86_64.whl
```

Verify the effective binary as described above. PyPTO's flat-2-D NZ support
requires PTOAS to accept a rank-5 `layout=nz` view; PyPTO commit `2eb7d3f8`
emits the implicit outer axis `E = 1` that PTOAS requires. Earlier commits fail
compilation with:

```text
'pto.make_tensor_view' op user-specified layout=nz requires a rank-5 view, got rank 4
```

## 4. Build the runtime

The validated runtime branch adds a CANN 9.3 compatibility fix: GCC 13 rejects
`memset` on a non-trivial type with `-Werror=class-memaccess`, so five sites cast
the destination to `void *`:

```text
src/common/host_build_graph/shared/runtime_init.cpp:255
src/a2a3/runtime/tensormap_and_ringbuffer/runtime/shared/runtime_init.cpp:579,605
src/a5/runtime/tensormap_and_ringbuffer/runtime/shared/runtime_init.cpp:614,640
```

That fix is commit `7186788f` on `env/a5-cann93-buildfix`. Build with:

```bash
cd <a5-root>/pypto/runtime
uv pip install -e . --python <a5-root>/venvs/pypto/bin/python
```

When the runtime source revision changes, a plain editable install may reuse the
previously built extension. Force a rebuild with:

```bash
uv pip install --reinstall -e . --python <a5-root>/venvs/pypto/bin/python
```

The runtime's Python layer checks the compiled extension against the source
revision on import. If it reports:

```text
_task_interface was built from <old>, but this source tree is at <new>
```

the extension is stale and the rebuild above is required. The `--no-build-isolation`
form is not usable in this venv unless `scikit_build_core` is installed into it;
the default isolated build works.

## 5. Build PyPTO

```bash
bash <a5-root>/build_pypto.sh
```

The script activates the same toolchain/venv/torch/CANN environment and runs:

```bash
uv pip install -e . --no-deps --python <a5-root>/venvs/pypto/bin/python
```

Only C++ changes require a rebuild; Python edits in the editable install take
effect immediately. A header-only change still rebuilds the extension and was
observed at about one minute on this machine with ccache warm.

## 6. Smoke test through task-submit

NPU work must go through the submission wrapper. It allocates a device, exports
`TASK_DEVICE`, and appends `--device <id>` to the inner command:

```bash
task-submit --device auto --max-time 0 --timeout 0 \
    --run <a5-root>/smoke/run_smoke.sh
```

`run_smoke.sh` activates the environment and runs `pypto_smoke.py`, which checks
both the simulator and the real device. Expected lines:

```text
OK smoke platform=a5sim device=0
OK smoke platform=a5 device=0
```

The NPU lock is released when the task exits. Task output is written under the
submission tool's task directory; a long task can be reattached with:

```bash
task-submit --wait <task-id>
```

## 7. Run the 10g GMM case

Compile-only check (host side, no NPU):

```bash
python <a5-root>/smoke/compile_10g.py
```

Expected: `COMPILE_OK <artifact-dir>`.

Profile run on the device:

```bash
task-submit --device auto --max-time 0 --timeout 0 \
    --run <a5-root>/smoke/run_gmm_10g_profile.sh
```

The script runs
`examples/advanced/10g_group_matmul_i8_flat_nz_unitflag.py -p a5 -d "$TASK_DEVICE" --profile`.
The example verifies bit-exactness against an FP64 reference, writes
`chip_swimlane_records.json`, `pmu.csv`, and `deps.json` under
`build_output/gmm_flat_nz_unitflag_profile/dfx_outputs/`, and prints the
`swimlane_summary.py` command for the makespan.

## AscendC INT8 reference

CANN 9.3 already installs the grouped-matmul aclnn operator
(`aclnnGroupedMatmulWeightNz`, `aclnnGroupedMatmulV5`; `libopapi.so` exports
both), so no ops-transformer build is needed. `scripts/bench/ascendc_gmm_int8_bench.cpp`
drives it with the same shape, dtype and per-group NZ layout as the PyPTO 10k
kernel:

```text
x          [8192, 3584] INT8 ND
weight     [32, 3584, 3072] INT8, per-group NZ [32, 96, 224, 16, 32]
out        [8192, 3072] INT32
groupType  0, groupList cumulative, splitItem 3
```

Build and run `scripts/bench/run_ascendc_gmm_int8.sh` through `task-submit`.
The per-group NZ blocks are converted separately (one 352 MB TransData faults)
and concatenated into one stacked device tensor; the GMM op receives a weight
list of one.

Measured on the 28-AIC board:

| Metric | Value |
| ------ | ----- |
| AscendC msprof Task Duration | 286.6 / 288.4 / 290.9 us (Block Dim 28, cube) |
| AscendC ACL event around a warmed launch | 322.0–323.4 us |
| PyPTO 10k AICore makespan | 332.8 us |
| PyPTO 10k PMU cube / MTE2 | 240.6 / 313.9 us per core |
| AscendC PMU cube / MTE2 | 243.3 / 269.0 us per core |

Cube is already matched. The remaining ~15% is MTE2: both kernels are ~97%
MTE2-busy and move the same ~25.1 MB per core, but AscendC sustains ~89 GB/s
versus PyPTO's ~82 GB/s. AscendC's fixpipe (111 us) is overlapped under MTE2;
PyPTO's is only 25 us, so unit-flag overlap is not the missing piece.

### Pipeline trace and overlap

`msprof op` can emit a Chrome `trace.json`; the flag is
`--aic-metrics=PipeTimeline,PipeUtilization` (there is no
`--aicmetric=pipeline`). For the AscendC GMM, the trace for core 0 shows:

```text
MTE2   busy 298.6 us / 305.4 us horizon (97.8%)
CUBE   busy 263.4 us (86.3%); 260.7 us of it overlaps MTE2
FIXP   busy 132.0 us (43.2%); 127.8 us of it overlaps MTE2
```

That is the "cube/MTE/fixpipe fully covered" shape: MTE2 is the long pole and
the other pipes run underneath it.

The PyPTO runtime kernel (`aicore_kernel_0_mix_aic`) cannot be profiled with
`msprof op` in this environment: kernel replay reports
`Kernel run on device ... failed`, and application replay raises an AICPU
exception / scheduler timeout (507018, `S1:running-stalled`). Use PyPTO's own
DFX instead (`enable_chip_swimlane=4`, `enable_pmu=2`, then
`swimlane_summary.py`). For 10k the PMU numbers are cube 240.6us, MTE2
313.9us, fixpipe 25.3us and AICore makespan 332.8us. If cube and MTE2 were
serialized the kernel could not finish in 332.8us (they sum to 554.5us), so at
least 221us of cube runs under MTE2. MTE2 occupies 94% of the makespan, while
fixpipe is small and not limiting.

`msprof op` profiles only the first kernel launch by default. The harness runs
32 weight TransData conversions, one GMM warm-up, then five measured GMM calls.
Skip 33 launches to capture a warmed GMM kernel:

```bash
msprof op --output=./msprof_gmm --launch-skip-before-match 33 \
    --launch-count 1 --warm-up 0 ./ascendc_gmm_int8_bench
```

The 220 us FP8 figure in the project notes is a different dtype on a 32-AIC
Ascend950DT_9582; it is not the INT8 target for this board. The FP8 case and
the bandwidth limit behind that figure are covered below.

### FP8 no-scale case and the 220 us baseline

`examples/advanced/10l_group_matmul_fp8_pergroup_nz_balanced.py` runs 10k's
schedule with FP8E4M3FN inputs and an FP16 output, matching the user's AscendC
dtype and per-group NZ layout. The A5 Cube accumulates FP8 in FP32 through the
plain `TMATMUL` (no MX scale); keep the accumulator FP32 and let the final
Acc->GM store narrow to FP16. Do not add an explicit
`pl.cast(acc, pl.FP16, mode="rint")` before that store: the autotiler folds
that cast only into a *later matmul's* Acc->Mat writeback, not into an Acc->GM
store, so it stays a 256x256 Vector `tcvt` and overflows UB with 640 KB of Vec
staging.

Measured in one task (device 1) with `PYPTO_PTOAS_SYNC=bufid`, 3 warmup + 10
measured launches:

| Kernel | Data path | benchmark `device_us` | profile AICore makespan | cube | MTE2 | fixpipe |
| ------ | --------- | --------------------- | ----------------------- | ---- | ---- | ------- |
| 10k INT8 | INT8 -> INT32 | 365.8 us | 332.8 us | 240.6 us | 313.9 us | 25.3 us |
| 10l FP8 | E4M3 -> FP16 | 335.7 us | 302.3 us | 239.5 us | 282.8 us | 13.3 us |
| AscendC INT8 | INT8 -> INT32 | ACL 322-323 us | msprof 286.6-290.9 us | 243.3 us | 269.0 us | 111.0 us |

The PMU figures are per core at 1.65 GHz and both PyPTO kernels load 25.1 MB
per core from GM into L1. FP8 raises MTE2 from 80 to 89 GB/s, exactly
AscendC's INT8 rate on this board, and is 96.6% MTE2-busy with 96.9% core
busy. Cube + MTE2 sum to 522.3 us against a 302.3 us makespan, so at least
220 us of cube runs under MTE2; fixpipe is only 13.3 us. That is the same
MTE2-bound shape as the AscendC trace. The 30 us gain over INT8 is a 31.1 us
MTE2 reduction (313.9 -> 282.8 us); fixpipe also halves (25.3 -> 13.3 us) but
was already under MTE2, and cube is unchanged. FP16 halves the output bytes.
At the kernel level the FP8 case
is within 5.5% of the AscendC INT8 library on the same board, even though
that library writes an INT32 output (twice the bytes).

CANN cannot produce this case through the library operator. The 950 checker
(`grouped_matmul_950_checker.cpp`, `CheckBasicQuantParams` and
`CheckWeightNzSpecialParams`) requires a non-null scale for FP8 whenever the
output is not INT32, and with a FRACTAL_NZ weight and E4M3 inputs it then
requires both `scale` and `perTokenScale` to be FLOAT8_E8M0 -- the MX path.
A plain FP8 matmul with a FLOAT32 unit scale and no perTokenScale is rejected
with `161002`. A same-board AscendC number for the user's custom no-scale FP8
kernel therefore needs that kernel's source or binary.

Reproduce the benchmark and the profile with:

```bash
PYPTO_PTOAS_SYNC=bufid python examples/advanced/10l_group_matmul_fp8_pergroup_nz_balanced.py \
    -p a5 -d "$TASK_DEVICE" --rounds 10
PYPTO_PTOAS_SYNC=bufid python examples/advanced/10l_group_matmul_fp8_pergroup_nz_balanced.py \
    -p a5 -d "$TASK_DEVICE" --profile
```

#### Why 220 us is not reachable on this board

Core count is not the whole story. Each of the 384 output tiles loads 0.917 MB
of activations and 0.917 MB of weight, for 704.6 MB of L1 loads per launch.
The 12 N tiles of a group share its activations, so only 29.4 MB of those
bytes are unique; each group's `[K, N]` weight is used by that group alone and
all 352.3 MB is compulsory. The FP16 output adds 50.3 MB of writes. The user's
220 us on the 32-AIC board therefore needs at least 381.7 MB / 220 us =
1.74 TB/s of HBM reads. Even scaling that number perfectly to 28 cores
(251 us) needs 1.52 TB/s.

This board is far below that. `scripts/bench/mte2_bandwidth` streams 1 GiB of
unique GM data per launch with 8-deep 64 KiB multibuffering:

| Blocks | Aggregate | Per core |
| ------ | --------- | -------- |
| 28 | 1.11 TB/s | 37-40 GB/s |
| 14 | 1.08 TB/s | 77 GB/s |
| 1 | 165 GB/s | 165 GB/s |

The FP8 GMM's compulsory reads over its 302.3 us makespan are already
381.7 MB / 302.3 us = 1.26 TB/s, above the pure streaming probe, so the probe
is a conservative floor for this board. The kernel is at the memory floor;
there is no remaining pipeline-overlap gap of the size implied by 220 us.
The 32-AIC 950DT must be supplying roughly 1.7-2.0 TB/s for that number.

The decisive same-board check is the streaming probe on the 32-core machine:

```bash
SOC_VERSION=ascend950dt_9582 bash scripts/bench/mte2_bandwidth/build.sh
task-submit --device auto --max-time 0 --timeout 0 \
    --run "bash scripts/bench/mte2_bandwidth/run.sh 32"
```

If it reports ~1.7-2.0 TB/s, the 220 us figure is explained by the board. If
it reports ~1.1 TB/s like the 950PR, the user's kernel is doing something the
library cannot express and its source is required.

## Core count

All eight whitelisted devices (0-7) report **28 AIC at 1.65 GHz**. The job
submission layer may place a task on any of them, but the visible count is 28.
The 32-AIC / 220us AscendC baseline in the project notes was recorded on an
Ascend950DT_9582, which is a different board: do not compare makespans across
the two without normalizing for the core count. Core-count normalization alone
(28/32) makes 220us equivalent to 251us here, but the FP8 memory-floor analysis
above shows the 950DT's HBM bandwidth is the larger factor; run the streaming
probe on that board to separate the two.

A 32-block SPMD launch on this board runs 28 blocks in the first wave and 4 in
a second wave; the makespan then includes two full block latencies and
`core_busy_utilization_pct` is low. Read the count from the swimlane record
rather than assuming, and size the grid with
`pl.system.available_cluster_count()`:

```bash
python examples/utils/swimlane_summary.py <record>
# record: ... core_types: ['aic', ...]
```

`npu-smi` is only usable inside a `task-submit` task (the submission adds the
`HwHiAiUser` group); it fails with `-8005` from a plain SSH shell:

```bash
task-submit --device 1 --run /path/to/a5_coreinfo.sh
# npu-smi info -t common -i "$TASK_DEVICE"
#   Aicore Count : 28
#   Aicore Freq(MHZ) : 1650
```

## Known limitations observed

- With the default event synchronization, the static `AccPhase` / `STPhase`
  attributes reach the generated code (`TMATMUL_ACC<AccPhase::Final>`,
  `TSTORE<STPhase::Final>`) but PTOAS 0.67's sync inserter still emits the
  software `set_flag(PIPE_M, PIPE_FIX) / wait_flag(PIPE_M, PIPE_FIX) /
  set_flag(PIPE_FIX, PIPE_M)` pair around the store. Unit flags alone therefore
  do not remove the cube/fixpipe serialization on that path.
- `PYPTO_PTOAS_SYNC=bufid` selects PTOAS's A5 buffer-id sync pass instead. It
  replaces the M->FIX event pair with `get_buf(PIPE_FIX, ...)` / `rls_buf`, and
  the balanced 10h kernel measured 354.6us AICore makespan at 97.7% core busy
  (versus 376.4us with the event path), bit-exact. The balanced AutoTile
  variant (10i) measured 331.7-338.5us at 97.2% core busy across two runs with
  the same option. The
  setting is A5-only and opt-in because the PTOAS bufid pass does not convert
  every GM dependency.
- The 10g example peels the first and last K chunks out of the pipeline, so per
  N tile two matmul/load phases run without the flat pipeline's overlap. Keep
  this in mind when comparing its cube time with a fully pipelined variant.

## Security

The shared machine is accessed with an SSH key. Only the public half belongs in
the repository (`scripts/keys/`); the private key stays on the workstation and
must never be copied to the shared machine or committed.
