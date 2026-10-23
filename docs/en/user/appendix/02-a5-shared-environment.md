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
| Device | Ascend 950DT_9582 (A5) |
| Nominal AI cores | 32 AIC (the scheduler may expose fewer; see [Core count](#core-count)) |
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

## Core count

The nominal part has 32 AIC, but the job submission layer may place a task on a
die or partition with fewer visible cores. Read the count from the swimlane
record rather than assuming:

```bash
python examples/utils/swimlane_summary.py <record>
# record: ... core_types: ['aic', ...]
```

A 32-block SPMD launch on a 28-AIC placement runs 28 blocks in the first wave
and 4 in a second wave; the makespan then includes two full block latencies and
`core_busy_utilization_pct` is low. Compare makespans only between runs with the
same visible core count, or make the grid match the placement.

## Known limitations observed

- The static `AccPhase` / `STPhase` attributes reach the generated code
  (`TMATMUL_ACC<AccPhase::Final>`, `TSTORE<STPhase::Final>`) but PTOAS 0.67's
  sync inserter still emits the software
  `set_flag(PIPE_M, PIPE_FIX) / wait_flag(PIPE_M, PIPE_FIX) / set_flag(PIPE_FIX, PIPE_M)`
  pair around the store. Unit flags alone therefore do not currently remove the
  cube/fixpipe serialization in this environment.
- The 10g example peels the first and last K chunks out of the pipeline, so per
  N tile two matmul/load phases run without the flat pipeline's overlap. Keep
  this in mind when comparing its cube time with a fully pipelined variant.

## Security

The shared machine is accessed with an SSH key. Only the public half belongs in
the repository (`scripts/keys/`); the private key stays on the workstation and
must never be copied to the shared machine or committed.
