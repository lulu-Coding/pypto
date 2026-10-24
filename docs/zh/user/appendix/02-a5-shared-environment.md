# 共享 A5 测试环境

本附录记录一套可工作的 PyPTO 环境搭建流程：目标机器是位于作业提交封装之后的共享
Ascend A5 测试机。它用于复现环境、以及解读该机器上跑出的性能数据，并不是所有
环境通用的官方安装路径。

尖括号中的值是站点相关的：`<a5-host>`、`<ssh-port>`、`<user>`，以及共享机上的
工作根目录 `<a5-root>`（下文用 `/home/pypto-tools`、`/home/pyptouser/luxiaolu`
举例）。

## 目标环境

| 项目 | 值 |
| ---- | -- |
| 设备 | Ascend950PR 9579（A5），板卡 A310-50-C00MM304A1；`npu-smi info -t board` |
| 实测 AI Core | 8 张白名单设备均为 28 AIC @ 1.65 GHz；`npu-smi info -t common` |
| 系统 / 编译器 | Ubuntu 24.04，系统 GCC 13.3.0 |
| CANN | 9.3.0，`/usr/local/Ascend/ascend-toolkit` |
| NPU 访问 | 共享机；所有 NPU 工作通过 `task-submit`，不要直接跑 `python` |

PyPTO toolchain 自带 GCC 15.2.0（`gcc-15` / `g++-15`），但其 `bin` 目录没有不带
版本后缀的 `g++`，因此 CMake 实际仍使用系统的 `/usr/bin/g++`（13.3.0）。验证过的
构建使用的就是这个编译器。

## 目录结构

```text
<a5-root>/
├── pypto/                       # 主工程，分支 group-matmul-i8
│   ├── runtime/                 # runtime，分支 env/a5-cann93-buildfix
│   ├── 3rdparty/libbacktrace/
│   ├── 3rdparty/msgpack-c/
│   └── ...
├── venvs/pypto/                 # 虚拟环境
├── build_runtime.sh
├── build_pypto.sh
├── ptoas-0.60-cp310-cp310-manylinux_2_34_x86_64.whl
├── pypto-prefetch-extra/        # prefetch 补丁和未跟踪文件
└── smoke/
    ├── pypto_smoke.py
    ├── run_smoke.sh
    ├── run_gmm_10g_profile.sh
    └── compile_10g.py
```

## 软件包

| 组件 | 版本 / pin | 获取方式 |
| ---- | ---------- | -------- |
| PyPTO toolchain | 2026.08.5 | 预装在 `/opt/pypto/toolchain`；`pypto-toolchain export 2026.08.5` 激活 |
| Python | 3.10.18 | toolchain |
| uv | 0.9.28 | toolchain |
| ccache | 4.10.2 | toolchain |
| CMake / Ninja | 系统 | 预装 |
| CANN | 9.3.0 | 预装于 `/usr/local/Ascend/ascend-toolkit` |
| PTOAS | v0.60 wheel | GitHub release 资产，经 `gh-proxy.com` 下载；SHA256 `3a3f428a12b4f0a1bc61f978fd675417da543bab09494344426bb35977de842a` |
| PyPTO | 分支 `group-matmul-i8` @ `2eb7d3f8` | git bundle |
| runtime（`simpler`） | 分支 `env/a5-cann93-buildfix` @ `7186788f` | git bundle |
| pto-isa | `5a4f74cb` | git bundle |
| libbacktrace | `6f8310e` | git bundle |
| msgpack-c | `91990874` | git bundle |
| torch | 2.6.0+cpu | 共享目录 `/home/pypto-tools/python-packages/torch-cpu-2.6.0`，通过 `activate.sh` 激活 |
| numpy | 2.2.6 | uv + 阿里云 PyPI 镜像 |
| pytest | 9.1.1 | uv + 阿里云 PyPI 镜像 |
| cloudpickle | 3.1.2 | uv + 阿里云 PyPI 镜像 |
| PyYAML | 6.0.3 | uv + 阿里云 PyPI 镜像 |

PyPI 上的 `ptoas` 只有 0.1.x，pin 住的版本必须从项目的 GitHub release 获取。该机器
上 CANN 的 `set_env.sh` 会把自身的 `bin` 和 `python/site-packages` 前置，因此
`ptoas` 实际解析到 CANN 自带的二进制，它报告 `ptoas 0.67`。pip wheel 仍保留，以便
在不被 CANN 覆盖的环境中使用。用 `command -v ptoas && ptoas --version` 确认实际
生效的二进制，不要根据 wheel 标签推断。

## Shell 环境

每次构建或运行都从同一套环境开始：

```bash
eval "$(pypto-toolchain export 2026.08.5)"
source <a5-root>/venvs/pypto/bin/activate
source /home/pypto-tools/python-packages/torch-cpu-2.6.0/activate.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

安装任何东西前先把 `UV_INDEX_URL` 指向可访问的镜像。该机器上验证可用的是阿里云：

```bash
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export UV_CONSTRAINT=<a5-root>/pypto/build-constraints.txt
```

## 1. 传输源码

共享机可以访问 fork 的 bundle，但未必能访问 GitHub。在工作站生成 bundle，再用 SSH
拷过去：

```bash
git bundle create pypto.bundle group-matmul-i8
git bundle create runtime.bundle feat/a5-sdma-prefetch
git bundle create libbacktrace.bundle <pin>
git bundle create msgpack-c.bundle <pin>
git bundle create pto-isa.bundle <pin>

scp -P <ssh-port> *.bundle <user>@<a5-host>:<a5-root>/pypto-transfer/bundles/
```

在共享机上：

```bash
git clone -b group-matmul-i8 <a5-root>/pypto-transfer/bundles/pypto.bundle <a5-root>/pypto
cd <a5-root>/pypto
git clone -b env/a5-cann93-buildfix <a5-root>/pypto-transfer/bundles/runtime.bundle runtime
git clone <a5-root>/pypto-transfer/bundles/libbacktrace.bundle 3rdparty/libbacktrace
git clone <a5-root>/pypto-transfer/bundles/msgpack-c.bundle 3rdparty/msgpack-c
```

共享克隆的 `origin` 也指向这些 bundle。有新提交后的更新流程是：重新生成
`pypto.bundle`，覆盖旧文件，然后：

```bash
cd <a5-root>/pypto
git fetch origin group-matmul-i8
git pull --ff-only origin group-matmul-i8
```

submodule 同样因连通性问题手工克隆，因此 `git submodule status` 会显示前缀 `-`，
这是预期现象。**不要**运行 `git submodule update`——它会访问上游 URL 并失败。

## 2. 创建虚拟环境并安装 Python 包

```bash
/opt/pypto/toolchain/2026.08.5/python/bin/python3.10 -m venv <a5-root>/venvs/pypto
source <a5-root>/venvs/pypto/bin/activate

uv pip install --python <a5-root>/venvs/pypto/bin/python \
    numpy pytest cloudpickle pyyaml
```

torch 使用共享的 CPU 目录，不要装进 venv；默认 wheel 会拉入约 2 GB 本工作流用不到
的 CUDA 依赖。激活脚本会为共享目录设置 `PYTHONPATH`：

```bash
source /home/pypto-tools/python-packages/torch-cpu-2.6.0/activate.sh
```

## 3. 安装 PTOAS

```bash
uv pip install --no-deps --python <a5-root>/venvs/pypto/bin/python \
    <a5-root>/ptoas-0.60-cp310-cp310-manylinux_2_34_x86_64.whl
```

按上文确认实际生效的二进制。PyPTO 的二维平铺 NZ 支持要求 PTOAS 接受 rank-5 的
`layout=nz` view；PyPTO 提交 `2eb7d3f8` 会补上 PTOAS 要求的隐式外层轴 `E = 1`。
更早的提交编译时会失败：

```text
'pto.make_tensor_view' op user-specified layout=nz requires a rank-5 view, got rank 4
```

## 4. 构建 runtime

验证过的 runtime 分支包含一个 CANN 9.3 兼容修复：GCC 13 的
`-Werror=class-memaccess` 会拒绝在非平凡类型上调用 `memset`，因此把五处目标指针
强制转换为 `void *`：

```text
src/common/host_build_graph/shared/runtime_init.cpp:255
src/a2a3/runtime/tensormap_and_ringbuffer/runtime/shared/runtime_init.cpp:579,605
src/a5/runtime/tensormap_and_ringbuffer/runtime/shared/runtime_init.cpp:614,640
```

该修复是 `env/a5-cann93-buildfix` 上的提交 `7186788f`。构建命令：

```bash
cd <a5-root>/pypto/runtime
uv pip install -e . --python <a5-root>/venvs/pypto/bin/python
```

runtime 源码的 revision 变化后，普通 editable 安装可能复用已编译的扩展。强制重编：

```bash
uv pip install --reinstall -e . --python <a5-root>/venvs/pypto/bin/python
```

runtime 的 Python 层在 import 时会核对编译扩展与源码 revision。若报：

```text
_task_interface was built from <old>, but this source tree is at <new>
```

说明扩展过期，需要执行上面的重编。该 venv 中 `--no-build-isolation` 不可用，除非
另行安装 `scikit_build_core`；默认的隔离构建可以正常工作。

## 5. 构建 PyPTO

```bash
bash <a5-root>/build_pypto.sh
```

脚本会激活同一套 toolchain/venv/torch/CANN 环境并执行：

```bash
uv pip install -e . --no-deps --python <a5-root>/venvs/pypto/bin/python
```

只有 C++ 变更需要重编；editable 安装中的 Python 修改立即生效。仅头文件变更也会
触发扩展重编，在 ccache 已热的情况下本机实测约一分钟。

## 6. 通过 task-submit 做 smoke 测试

NPU 工作必须经过提交封装。它会分配设备、导出 `TASK_DEVICE`，并在内部命令后追加
`--device <id>`：

```bash
task-submit --device auto --max-time 0 --timeout 0 \
    --run <a5-root>/smoke/run_smoke.sh
```

`run_smoke.sh` 激活环境并运行 `pypto_smoke.py`，同时检查模拟器和真实设备。预期输出：

```text
OK smoke platform=a5sim device=0
OK smoke platform=a5 device=0
```

任务退出时会释放 NPU 锁。任务输出写在提交工具的任务目录下；长时间任务可以用以下
命令重新挂接：

```bash
task-submit --wait <task-id>
```

## 7. 运行 10g GMM 用例

仅编译检查（宿主机，不用 NPU）：

```bash
python <a5-root>/smoke/compile_10g.py
```

预期输出：`COMPILE_OK <artifact-dir>`。

设备上的 profile 运行：

```bash
task-submit --device auto --max-time 0 --timeout 0 \
    --run <a5-root>/smoke/run_gmm_10g_profile.sh
```

脚本执行
`examples/advanced/10g_group_matmul_i8_flat_nz_unitflag.py -p a5 -d "$TASK_DEVICE" --profile`。
用例会用 FP64 参考验证 bit-exact，并在
`build_output/gmm_flat_nz_unitflag_profile/dfx_outputs/` 下写出
`chip_swimlane_records.json`、`pmu.csv`、`deps.json`，同时打印用
`swimlane_summary.py` 查看 makespan 的命令。

## AscendC INT8 参考基准

CANN 9.3 已安装 grouped-matmul 的 aclnn 算子
（`aclnnGroupedMatmulWeightNz`、`aclnnGroupedMatmulV5`；`libopapi.so` 导出了
两者），不需要先编译 ops-transformer。`scripts/bench/ascendc_gmm_int8_bench.cpp`
用与 PyPTO 10k 相同的 shape、dtype 和 per-group NZ 布局调用它：

```text
x          [8192, 3584] INT8 ND
weight     [32, 3584, 3072] INT8, per-group NZ [32, 96, 224, 16, 32]
out        [8192, 3072] INT32
groupType  0, groupList 累积和, splitItem 3
```

通过 task-submit 运行 `scripts/bench/run_ascendc_gmm_int8.sh`。每组 NZ 单独转换
（一次性 352MB TransData 会触发向量核异常），再在 device 内拼成一个 stacked
tensor；GMM 算子收到长度为 1 的 weight list。

同一 28 核板卡实测：

| 指标 | 数值 |
| ---- | ---- |
| AscendC msprof Task Duration | 286.6 / 288.4 / 290.9 us（Block Dim 28，cube） |
| AscendC ACL event（预热后） | 322.0–323.4 us |
| PyPTO 10k AICore makespan | 332.8 us |
| PyPTO 10k PMU cube / MTE2 | 每核 240.6 / 313.9 us |
| AscendC PMU cube / MTE2 | 每核 243.3 / 269.0 us |

cube 已经打平。剩下的约 15% 在 MTE2：两边都约 97% MTE2 忙碌、每核搬运同样的
约 25.1MB，但 AscendC 约 89GB/s，PyPTO 约 82GB/s。AscendC 的 fixpipe（111us）
被 MTE2 覆盖；PyPTO 只有 25us，所以 unit-flag 重叠不是缺失的那一块。

### Pipeline trace 与重叠度

`msprof op` 可以导出 Chrome `trace.json`；参数是
`--aic-metrics=PipeTimeline,PipeUtilization`（没有 `--aicmetric=pipeline`）。
AscendC GMM 的 core 0 trace 为：

```text
MTE2   busy 298.6 us / horizon 305.4 us (97.8%)
CUBE   busy 263.4 us (86.3%)，其中 260.7 us 与 MTE2 重叠
FIXP   busy 132.0 us (43.2%)，其中 127.8 us 与 MTE2 重叠
```

这就是“cube/MTE/fixpipe 基本全覆盖”的形态：MTE2 是长板，其余 pipe 在它下面跑。

PyPTO runtime 的 kernel（`aicore_kernel_0_mix_aic`）在当前环境里不能用
`msprof op` 采集：kernel replay 报 `Kernel run on device ... failed`，
application replay 会触发 AICPU 异常/调度超时（507018，
`S1:running-stalled`）。PyPTO 侧应使用自带 DFX
（`enable_chip_swimlane=4`、`enable_pmu=2`，再跑 `swimlane_summary.py`）。10k 的
PMU 为 cube 240.6us、MTE2 313.9us、fixpipe 25.3us、AICore makespan 332.8us。
如果 cube 与 MTE2 串行，332.8us 内不可能完成（两者相加 554.5us），因此至少有
221us 的 cube 与 MTE2 重叠。MTE2 占 makespan 的 94%，fixpipe 很小，不是瓶颈。

`msprof op` 默认只 profile 第一次 kernel launch。该基准先做 32 次 weight
TransData 转换、1 次 GMM 预热，然后 5 次测量；跳过 33 次 launch 可抓到预热后的
GMM kernel：

```bash
msprof op --output=./msprof_gmm --launch-skip-before-match 33 \
    --launch-count 1 --warm-up 0 ./ascendc_gmm_int8_bench
```

项目笔记里的 220us 是 32 核 Ascend950DT_9582 上的 FP8 数据，不是这块板卡的
INT8 目标。FP8 用例以及该数字背后的带宽限制见下文。

### FP8 no-scale 用例与 220us 基线

`examples/advanced/10l_group_matmul_fp8_pergroup_nz_balanced.py` 用 10k 的调度
跑 FP8E4M3FN 输入、FP16 输出，与用户 AscendC 的 dtype 和 per-group NZ 布局一致。
A5 Cube 用普通 `TMATMUL` 以 FP32 累加 FP8（不涉及 MX scale）；累加器保持 FP32，
由最后一次 Acc->GM store 窄化为 FP16。不要在该 store 前显式写
`pl.cast(acc, pl.FP16, mode="rint")`：autotiler 只把它折进后续 matmul 的
Acc->Mat 写回，不会折进 Acc->GM store，结果会变成 256x256 的 Vector `tcvt`，
需要 640KB Vec 暂存并超出 UB。

同一个任务（device 1，`PYPTO_PTOAS_SYNC=bufid`，3 次 warmup + 10 次测量）：

| 内核 | 数据路径 | benchmark `device_us` | profile AICore makespan | cube | MTE2 | fixpipe |
| ---- | -------- | --------------------- | ----------------------- | ---- | ---- | ------- |
| 10k INT8 | INT8 -> INT32 | 365.8 us | 332.8 us | 240.6 us | 313.9 us | 25.3 us |
| 10l FP8 | E4M3 -> FP16 | 335.7 us | 302.3 us | 239.5 us | 282.8 us | 13.3 us |
| AscendC INT8 | INT8 -> INT32 | ACL 322-323 us | msprof 286.6-290.9 us | 243.3 us | 269.0 us | 111.0 us |

PMU 是 1.65GHz 下的单核数据，两个 PyPTO 内核都从 GM 搬 25.1MB/core 到 L1。
FP8 把 MTE2 从 80GB/s 提到 89GB/s，正好是本板 AscendC INT8 的速率；MTE2 占用
96.6%，core busy 96.9%。cube + MTE2 合计 522.3us，而 makespan 只有 302.3us，
因此至少有 220us 的 cube 在 MTE2 下运行；fixpipe 只有 13.3us。这与 AscendC
trace 的 MTE2-bound 形态一致。FP8 相对 INT8 的 30us 收益对应 MTE2 减少 31.1us
（313.9 -> 282.8us）；fixpipe 也减半（25.3 -> 13.3us），但它本就在 MTE2 之下，
cube 不变。FP16 使输出字节减半。在核内层面，FP8 与同板 AscendC INT8 库算子的
差距为 5.5%，而后者写的是字节数翻倍的 INT32 输出。

CANN 库算子无法复现该用例。950 的 checker
（`grouped_matmul_950_checker.cpp`，`CheckBasicQuantParams` 与
`CheckWeightNzSpecialParams`）要求：输出不是 INT32 时 FP8 必须提供非空 scale；
而当 weight 为 FRACTAL_NZ、输入为 E4M3 时，scale 和 perTokenScale 都必须是
FLOAT8_E8M0，也就是 MX 路径。带 FLOAT32 单位 scale、没有 perTokenScale 的普通
FP8 matmul 会报 `161002`。所以要在同板测用户那版 no-scale FP8 自定义 kernel，
需要它的源码或可执行文件。

复现 benchmark 和 profile：

```bash
PYPTO_PTOAS_SYNC=bufid python examples/advanced/10l_group_matmul_fp8_pergroup_nz_balanced.py \
    -p a5 -d "$TASK_DEVICE" --rounds 10
PYPTO_PTOAS_SYNC=bufid python examples/advanced/10l_group_matmul_fp8_pergroup_nz_balanced.py \
    -p a5 -d "$TASK_DEVICE" --profile
```

#### 为什么本板达不到 220us

核数不是全部差异。384 个输出 tile 各自从 L1 视角加载 0.917MB 激活和 0.917MB
weight，共 704.6MB；一个 group 的 12 个 N tile 共享激活，所以唯一激活只有
29.4MB；每个 group 的 `[K, N]` weight 只被该 group 使用，352.3MB 全部是必需的。
FP16 输出再增加 50.3MB 写入。因此用户 32 核板上的 220us 至少需要
381.7MB / 220us = 1.74TB/s 的 HBM 读；即使按核数完美折算到 28 核（251us），
也需要 1.52TB/s。

本板远低于这个值。`scripts/bench/mte2_bandwidth` 每次 launch 用 8 级 64KiB
multibuffer 流式读取 1GiB 唯一 GM 数据：

| blockDim | 聚合带宽 | 单核带宽 |
| -------- | -------- | -------- |
| 28 | 1.11 TB/s | 37-40 GB/s |
| 14 | 1.08 TB/s | 77 GB/s |
| 1 | 165 GB/s | 165 GB/s |

FP8 GMM 在 302.3us 内的必需读为 381.7MB / 302.3us = 1.26TB/s，高于纯流式探针，
所以探针是本板的保守下限。内核已经贴着存储下限；220us 所暗示的流水重叠差距
并不存在。要让 220us 成立，32 核 950DT 需要提供大约 1.7-2.0TB/s。

决定性的同板验证是在 32 核机器上跑同一个探针：

```bash
SOC_VERSION=ascend950dt_9582 bash scripts/bench/mte2_bandwidth/build.sh
task-submit --device auto --max-time 0 --timeout 0 \
    --run "bash scripts/bench/mte2_bandwidth/run.sh 32"
```

如果报告 ~1.7-2.0TB/s，220us 就是板卡差异；如果仍约 1.1TB/s，则用户的 kernel
做了库算子无法表达的事情，需要其源码。

## 核数

8 张白名单设备（0–7）都报告 **28 AIC @ 1.65 GHz**。作业提交层可能把任务放到任意
一张卡上，但可见核数是 28。项目笔记里的 32 AIC / 220us AscendC 基线是在
Ascend950DT_9582 上测的，那是另一块板卡；在按核数归一化之前，不要直接比较两者的
makespan。仅按核数折算（28/32）时 220us 相当于本板的 251us，但上面的 FP8
存储下限分析表明 950DT 的 HBM 带宽是更大的因素；需要在该板上跑流式探针来区分两者。

在这块板卡上，32 个 block 的 SPMD 启动第一波跑 28 个，第二波跑 4 个；makespan
因此包含两整轮 block 延迟，`core_busy_utilization_pct` 也会偏低。不要假设核数，
从 swimlane 记录读取，并用 `pl.system.available_cluster_count()` 决定网格宽度：

```bash
python examples/utils/swimlane_summary.py <record>
# record: ... core_types: ['aic', ...]
```

`npu-smi` 只能在 `task-submit` 任务内使用（提交时会补上 `HwHiAiUser` 组）；普通
SSH shell 里会以 `-8005` 失败：

```bash
task-submit --device 1 --run /path/to/a5_coreinfo.sh
# npu-smi info -t common -i "$TASK_DEVICE"
#   Aicore Count : 28
#   Aicore Freq(MHZ) : 1650
```

## 已观察到的限制

- 默认的 event 同步模式下，静态 `AccPhase` / `STPhase` 属性确实进入了生成代码
  （`TMATMUL_ACC<AccPhase::Final>`、`TSTORE<STPhase::Final>`），但 PTOAS 0.67 的
  同步插入仍会在 store 前后生成软件
  `set_flag(PIPE_M, PIPE_FIX) / wait_flag(PIPE_M, PIPE_FIX) / set_flag(PIPE_FIX, PIPE_M)`
  事件对。因此在这条路径上 unit flag 本身还不能消除 cube 与 fixpipe 的串行化。
- `PYPTO_PTOAS_SYNC=bufid` 会选择 PTOAS 的 A5 buffer-id 同步 pass。它把 M→FIX
  事件对替换成 `get_buf(PIPE_FIX, ...)` / `rls_buf`，均衡化后的 10h 用例在该模式下
  实测 AICore makespan 354.6us、core busy 97.7%（event 路径为 376.4us），且
  bit-exact。均衡化的 AutoTile 版本（10i）在同样选项下两次运行实测
  331.7–338.5us、core busy 97.2%。该选项仅适用于 A5 且为显式开启，因为 PTOAS
  的 bufid pass 并不转换所有 GM 依赖。
- 10g 用例把第一个和最后一个 K chunk 都从流水线中剥出，因此每个 N tile 有两段
  matmul/load 无法享受 flat pipeline 的重叠。将它和全流水变体的 cube 时间比较时
  需要留意。

## 安全

共享机通过 SSH 密钥访问。仓库中只应出现公钥（`scripts/keys/`）；私钥保留在工作站，
绝不能拷贝到共享机，也不能提交。
