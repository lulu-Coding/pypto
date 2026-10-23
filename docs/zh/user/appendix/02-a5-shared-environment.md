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
| 设备 | Ascend 950DT_9582（A5） |
| 标称 AI Core | 32 AIC（调度器可能只暴露更少，见[核数](#核数)） |
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

## 核数

器件标称 32 AIC，但作业提交层可能把任务放到只暴露更少核的 die 或分区上。不要假设，
从 swimlane 记录中读取实际数量：

```bash
python examples/utils/swimlane_summary.py <record>
# record: ... core_types: ['aic', ...]
```

32 个 block 的 SPMD 启动落在 28 AIC 的放置上时，第一波跑 28 个 block，第二波跑
4 个；makespan 因此包含两整轮 block 延迟，`core_busy_utilization_pct` 也会偏低。
只有在可见核数相同的两次运行之间比较 makespan，或者让网格匹配实际放置。

## 已观察到的限制

- 默认的 event 同步模式下，静态 `AccPhase` / `STPhase` 属性确实进入了生成代码
  （`TMATMUL_ACC<AccPhase::Final>`、`TSTORE<STPhase::Final>`），但 PTOAS 0.67 的
  同步插入仍会在 store 前后生成软件
  `set_flag(PIPE_M, PIPE_FIX) / wait_flag(PIPE_M, PIPE_FIX) / set_flag(PIPE_FIX, PIPE_M)`
  事件对。因此在这条路径上 unit flag 本身还不能消除 cube 与 fixpipe 的串行化。
- `PYPTO_PTOAS_SYNC=bufid` 会选择 PTOAS 的 A5 buffer-id 同步 pass。它把 M→FIX
  事件对替换成 `get_buf(PIPE_FIX, ...)` / `rls_buf`，均衡化后的 10h 用例在该模式下
  实测 AICore makespan 354.6us、core busy 97.7%（event 路径为 376.4us），且
  bit-exact。该选项仅适用于 A5 且为显式开启，因为 PTOAS 的 bufid pass 并不转换
  所有 GM 依赖。
- 10g 用例把第一个和最后一个 K chunk 都从流水线中剥出，因此每个 N tile 有两段
  matmul/load 无法享受 flat pipeline 的重叠。将它和全流水变体的 cube 时间比较时
  需要留意。

## 安全

共享机通过 SSH 密钥访问。仓库中只应出现公钥（`scripts/keys/`）；私钥保留在工作站，
绝不能拷贝到共享机，也不能提交。
