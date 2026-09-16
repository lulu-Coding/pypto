# SHMEM 三条 CMO 预取通路在 PyPTO 中的实现与前端语义设计

> **状态**：设计提案（RFC）
> **日期**：2026-09-16
> **目标设备**：Ascend950（A5）/ Ascend910B（A2A3）
> **关联文档**：`docs/zh/dev/l2-buffer-managed-design.md`（L2Buffer RFC）、`docs/zh/dev/l2-buffer-a5-cmo-path.md`（早期路径选型）、`docs/zh/dev/a5-sdma-prefetch-minimal-guide.md`（A5 使能最小方案）
> **来源**：SHMEM PR #459（A5 CMO/SDMA）完整链路提取分析，详见 `docs/zh/dev/a5-shmem-prefetch-extraction.md`（三条通路接口、调用栈、SQE 字段、供给链与最小可用程序）

---

## 0. 结论速览

SHMEM 在 A5 上的三条 GM→L2 预取通路，映射到 PyPTO 的状态如下：

| SHMEM 通路 | 执行位置 | PyPTO 前端语义 | 状态 |
|------------|----------|----------------|------|
| **A. Host 发起**：`aclrtCmoAsync(ptr, size, ACL_RT_CMO_TYPE_PREFETCH, stream)` | Host CPU（流序） | `pl.prefetch.host_async(x)`（Orchestration 函数内，**本文档 §5 设计**） | ❌ 未实现 |
| **B. Device 单 QP**：`aclshmemx_cmo_nbi` + `aclshmemx_sdma_quiet`（QP0，单 AIV） | AIV 内 | `pl.prefetch.make_context()` + `async_prefetch(x, ctx)` + `wait(evt, sess)`（单 AIV InCore kernel 内） | ✅ 已合入 main（#2089） |
| **C. Device 多 QP**：`aclshmemx_cmo_qp_nbi` + `aclshmemx_sdma_qp_quiet`（`qp_idx = GetBlockIdx()`） | 全部 AIV 并行 | 同上（多 AIV SPMD InCore kernel 内，channel group 自动按 AIV 划分） | ✅ 已合入 main（#2089）+ A5 使能进行中 |

核心结论：

1. **路径 B/C 不需要新的前端语义**。PR #2089 合入的 `pl.prefetch.*` op 家族已经以
   PyPTO 的标准分层方式（DSL→IR→PTO Codegen→PTOAS→pto-isa→STARS v2 直驱）实现了设备侧
   CMO 预取，且默认语义恰好就是 SHMEM 推荐的多 QP 模式（每 AIV 独立 channel group）。
   路径 B 是路径 C 在单 AIV kernel 上的自然退化。
2. **A5 上路径 B/C 打通只差运行时使能**（simpler workspace 供给 patch、pto-isa SQE 修复、
   ST 测试平台扩展），属进行中工作（§3.2），不需要架构变更。
3. **唯一缺失的是路径 A**（Host/编排层发起的流序预取）。本文档 §5 给出符合 PyPTO 架构的
   完整设计：新增 orchestration-legal 的 `prefetch.host_async` op。
4. 早期文档 `l2-buffer-a5-cmo-path.md` 中的 "kernel_entry wrapper 注入" 方案已被 #2089 的
   正式 IR op 路线取代（该文档 §9.3 的"阶段 2：PTO-ISA 新 op"已落地），wrapper 注入不再需要。

---

## 1. 输入：SHMEM 三条通路摘要（自包含）

以下事实来自 SHMEM 仓库（master @ 73064fa，PR #459）的提取分析，是本设计的输入。

### 1.1 三条通路

| 通路 | API | 特点 |
|------|-----|------|
| A. Host | `aclrtCmoAsync(dev_ptr, size, ACL_RT_CMO_TYPE_PREFETCH, stream)` | CANN runtime API；流序（kernel 在同 stream 后发即自动等待预取完成）；无 SHMEM 依赖；延迟高（host→device 往返），无并发 |
| B. Device 单 QP | `aclshmemx_cmo_nbi<T>(src, elem_size, CMO_TYPE_PREFETCH, ub_buf, ub_size, sync_id)` + `aclshmemx_sdma_quiet<T>(...)` | 仅 AIV 0（QP0）；设备侧执行；SHMEM 供给 STARS workspace |
| C. Device 多 QP | `aclshmemx_cmo_qp_nbi<T>(..., qp_idx, sync_id)` + `aclshmemx_sdma_qp_quiet<T>(..., qp_idx, ...)` | 每个 AIV 用 `qp_idx = GetBlockIdx()` 独立 QP，最多 72 并发；**SHMEM 实测最优** |

### 1.2 底层关键事实

- A5 上 CMO 仅支持 `CMO_TYPE_PREFETCH`（opcode=6），其余类型静默忽略；SDMA put/get 在
  950 后端直接 abort。
- 预取不改数据值，是纯 cache hint；`quiet` 通过 8B flag SDMA 写 SQE + GM 轮询实现完成等待。
- 设备侧提交 = 填 64B STARS v2 SDMA-CMO SQE（`wr_cqe=1, opcode=6, kernel_credit=254, qos=6,
  sssv/dssv/sns/dns=1`）→ DCCI 清缓存线 → 写 doorbell（3510 上偏移 0x0）。
- Host 供给链（SHMEM `aclshmemx_init_attr` SDMA 模式）：创建 STARS 流（`ACL_STREAM_DEVICE_USE_ONLY`）→
  分配 GM workspace（28KB：flag_info + channel_info[72] + notify 区域）→ AICPU 算子
  `AclnnShmemSdmaStarsQuery` 查询硬件并填充 channel info → 下发到设备全局变量。
- 约束：UB 临时缓冲 ≥64B 且 64B 对齐；CMO 仅在 AIV 执行；nbi 必须与同 QP 的 quiet 配对。

---

## 2. PyPTO 架构映射原则

本设计的所有取舍都遵循 PyPTO 既有分层与规则，逐条对应：

| 原则 | 出处 | 在本设计中的体现 |
|------|------|------------------|
| **两条 Codegen 路径**：设备语义走 InCore→PTO→PTOAS→pto-isa；编排语义走 TensorOp→Orchestration C++→simpler runtime | 架构（`docs/zh/dev/architecture-learning-guide.md` §1.3） | 路径 B/C 用 `prefetch.*` InCore op（已有）；路径 A 用 `prefetch.host_async` TensorOp + orchestration codegen（新设计） |
| **语义承载在 IR**：前端 API 构建标准 IR Call，不旁路 | add-op 技能 Phase A/B | `host_async` 走 `REGISTER_OP` + IR wrapper + DSL wrapper 标准三件套 |
| **运行时拥有执行策略**："The runtime owns the access policy" | `tensor.read`/`tensor.write` codegen 注释（`src/codegen/tensor_op_codegen.cpp:124-188`） | 生成的 AICPU 代码只调用 `rt_cmo_prefetch_async(tensor, bytes)`，地址解析、生产者依赖、CMO 落地方式由 simpler 决定 |
| **运行时供给经工件元数据握手**：`enable_sdma` → `ChipWorker(enable_sdma=True)` → `dma_workspace_provision` | `pto_backend.py` `_SDMA_WORKSPACE_OPS` / `worker.py` | 路径 A 不需要新握手（无 workspace），但沿用同一模式做能力门控（§5.6） |
| **跨层同步**：C++ op + Python 绑定 + 类型存根三处同步修改 | `.claude/rules/cross-layer-sync.md` | §5.7 的修改清单按此组织 |
| **Cache hint 语义不改数值** | #2089 的 ST 设计（非破坏性断言） | `host_async` 沿用同一测试范式（§5.8） |
| **文档先行 / en 为基准** | `.claude/rules/documentation.md` | §5.9 的文档清单 |

---

## 3. 现状盘点（截至 2026-09-16）

### 3.1 已合入 main：`pl.prefetch.*` 覆盖路径 B/C（PR #2089）

设备侧全链路已存在，且与 SHMEM 路径 C 的语义逐项对应：

```
DSL:   pl.prefetch.make_context() / async_prefetch(x, ctx) / session(ctx) / wait(evt, sess)
         (python/pypto/language/op/prefetch_ops.py, typing/prefetch_handle.py)
IR:    prefetch.make_context / prefetch.async_prefetch / prefetch.session / prefetch.wait
         (src/ir/op/prefetch/prefetch_async.cpp)
PTO:   pto.make_prefetch_async_context / pto.tprefetch_async /
       pto.get_prefetch_async_session / pto.comm.wait_async_event
         (src/backend/common/pto_ops_prefetch.cpp)
注入:  pto_backend.py 检测 prefetch.make_context → kernel_entry wrapper 注入
       __pypto_sdma_workspace = get_dma_workspace(args, DMA_WORKSPACE_SDMA)
       （pto_backend.py:679 _SDMA_WORKSPACE_OPS，:1058 sdma_setup）
握手:  kernel_config.py enable_sdma=True → worker.py ChipWorker(enable_sdma=True) →
       device_runner.py require_sdma → simpler dma_workspace_provision
ISA:   pto-isa TPREFETCH_ASYNC（include/pto/comm/async/sdma/TPrefetchAsyncImpl.hpp）
         → __sdma_cmo_prefetch（sdma_cmo_intrin.hpp）
         → AddOneCmoSqe（STARS v2 SDMA-CMO SQE 直驱 + doorbell）
ST:    tests/st/runtime/ops/test_prefetch_async.py（非破坏性断言 + 等待完成不挂死）
```

与 SHMEM 路径 B/C 的语义对应表：

| SHMEM（路径 B/C） | PyPTO/pto-isa | 说明 |
|---|---|---|
| `qp_idx = GetBlockIdx()`（每 AIV 独立 QP） | `BuildSdmaSession(..., kAutoChannelGroupIdx)` 默认解析为 `get_block_idx()`（`sdma_async_intrin.hpp:82-88`） | **默认语义相同**：多 AIV SPMD kernel 中每个 AIV 自动使用独立 channel group |
| 路径 B：QP0 单 AIV | 单 AIV InCore kernel 中调用（`get_block_idx()` 恒为 0） | 路径 B 是同一 op 在单 AIV kernel 上的退化，无需独立 API |
| `ub_buf`/`ub_size`（≥64B、64B 对齐 UB 临时缓冲） | `PrefetchAsyncContext::scratchTile`（`Vec` tile，`UB_ALIGN_SIZE` 对齐）+ `TASSIGN_IMPL(ctx.scratchTile, 0x0)` | 编译期分配，用户无感 |
| `aclshmemx_sdma_qp_quiet`（flag arm + 轮询） | `FinishSdmaPost`（事件完成记录）+ `pto.comm.wait_async_event`（`prefetch.wait`） | wait 语义显式化，比 SHMEM 的隐式配对更符合 IR 建模 |
| Host 供给：28KB workspace + AICPU StarsQuery + 72 channel | simpler `SdmaWorkspaceManager`（48 条 STARS 流 + 16KB workspace + `AclnnShmemSdmaStarsQuery`）+ `dma_workspace_provision` C API + `g_dma_workspace_addr` + `get_dma_workspace` intrinsic | 同一硬件契约（STARS v2），pto-isa 的供给更轻量 |
| 单 SQE 尺寸限制 | `kSingleSqeBlockBytes = 64MB`，`SdmaBaseConfig{64MB, 0, queue_num=1}`（`TPrefetchAsyncImpl.hpp:157-159`） | 超过 64MB 由 `SubmitCmoPrefetchSqes` 按 `queue_num` 拆分多 SQE |
| SQE 字段（opcode=6 等） | `AddOneCmoSqe`（`sdma_cmo_intrin.hpp:35-87`，A5 分支） | 与 SHMEM `aclshmemi_fill_stars_v2_cmo_sqe` 字节级等价（已修复 SQE 初始化差异） |
| `elem_size` 参数是元素数 | IR 侧约束 flat contiguous 1D（`CheckFlatContiguous1DSource`），字节量由 shape×dtype 推导 | PyPTO 前端比 SHMEM 更严格，避免误用 |

### 3.2 进行中：A5 使能三件套（不改架构）

| 项 | 内容 | 载体 | 状态 |
|----|------|------|------|
| pto-isa SQE 修复 | A5 STARS v2 CMO SQE 初始化（对齐 SHMEM 字节布局） | fork 分支 `fix/a5-stars-v2-sqe-init` @ `ebb1d62e` | 等离线验证通过后开上游 PR，之后 pypto bump `runtime/pto_isa.pin` |
| simpler workspace 供给 | `dma_workspace_supported_mask/channel_count/provision` 三 C 函数接通 a5 已有 `SdmaWorkspaceManager` | simpler patch（`src/a5/platform/onboard/host/comm_hccl.cpp`、`dma_workspace.h`） | 已实现并远程编译验证，待离线 E2E |
| ST 平台扩展 | `test_prefetch_async.py` 平台标记 `a2a3` → `a2a3, a5`（+sim no-op 契约说明） | pypto 工作树未提交改动 | 随上述两项落地后提交 |

---

## 4. 缺口分析：路径 A（Host/编排发起）

**语义需求**：在编排层（Orchestration 函数）预取一个 GM tensor 到 L2，使其先于"消费它的
InCore 任务"执行——即任务图级 latency hiding：预取与前置计算重叠，后续 kernel 的
`tile.load` 命中 L2。

**为什么不能复用现有 op**：`prefetch.async_prefetch` 是 InCore op（PTO codegen →
PTOAS → pto-isa 设备指令），在 Orchestration 函数中非法。Host 发起的 CMO（`aclrtCmoAsync`）
本质是编排层/运行时行为，必须走 TensorOp + orchestration codegen 路径——这正是 PyPTO
两条 codegen 路径的分工边界。

**为什么值得做**：与 SHMEM 的对比结论一致——设备侧路径（B/C）吞吐最优，但 host 侧路径
允许在**任务提交前**提前预热（例如上一轮计算仍在执行时，编排代码已为下一轮 tensor 发起
预取），且不占用 AIV 的 QP 深度。两者互补。

---

## 5. 路径 A 设计：`pl.prefetch.host_async`

### 5.1 前端语义

```python
import pypto.language as pl

@pl.program
class MyProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[N], pl.FP32], out: pl.Out[pl.Tensor[[N], pl.FP32]]):
        # tile.load 时命中 L2（预取已由编排层发起）
        t = pl.load(x, [0], [N])
        return pl.store(t, [0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(self, x, out):
        # Host/编排层发起 GM→L2 预取：cache hint，不改数据，无返回值。
        # 语义保证：本预取在随后提交的任务之前按执行流序生效。
        pl.prefetch.host_async(x)
        out = self.kernel(x, out)
        return out
```

语义定义：

- **非破坏性 cache hint**：与 `pl.prefetch.wait` 家族一致，不改变任何 tensor 值。
- **流序保证**：预取按执行流序先于"在本语句之后提交的任务"；用户不需要（也没有）wait。
- **依赖保证**：op 对源 tensor 声明 `ArgEffect::Read`，DSA/依赖分析保证它排在源 tensor 的
  生产者任务之后（地址已就绪才能发起）。
- **约束**：源必须是 flat contiguous 1D GM tensor（复用 `CheckFlatContiguous1DSource`，
  与 `prefetch.async_prefetch` 相同约束，因为底层同样只接受 `ptr + bytes`）。

### 5.2 层 B1：IR op（C++ 注册）

**文件**：`src/ir/op/prefetch/prefetch_async.cpp`（追加）

```cpp
// prefetch.host_async — orchestration-level host/stream-ordered GM->L2 prefetch.
//
// Semantics: issue an L2 prefetch for a flat contiguous 1D GM tensor from an
// orchestration function; the prefetch is stream-ordered before tasks
// submitted after this statement. The runtime owns the issuance policy (host
// aclrtCmoAsync / AICPU CMO SQE / simulator no-op). Changes no tensor values.
REGISTER_OP("prefetch.host_async")
    .set_op_category("TensorOp")
    .no_execution_memory_access()
    .set_description(
        "Issue an orchestration-level GM->L2 prefetch (cache hint) for a flat contiguous 1D "
        "tensor. The prefetch is ordered before subsequently submitted tasks on the execution "
        "stream. The runtime owns the issuance policy (host aclrtCmoAsync, AICPU CMO SQE, or "
        "simulator no-op). Changes no tensor values.")
    .add_argument("src", "Source tensor (TensorType, flat contiguous 1D GM)")
    // Prefetch needs the tensor's device address to be materialized, so it must
    // be ordered after the tensor's producer like any read.
    .set_arg_effect(0, ArgEffect::Read)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CheckArity("prefetch.host_async", "src", 1, args, kwargs);
      CheckFlatContiguous1DSource("prefetch.host_async", args[0]);
      return nullptr;  // statement-only op, 同 tensor.write 的 void 约定
    });
```

要点：

- 类别 `TensorOp` → 编排 codegen 自动接管（`IsTensorOp` 分发，见
  `src/codegen/orchestration/orchestration_codegen.cpp:1762`），InCore 函数中出现则被
  既有的 builtin/tensor 归属检查拒绝。
- 复用同文件已有的 `CheckArity` / `CheckFlatContiguous1DSource` 辅助函数，不新增校验设施。

### 5.3 层 B2/B3：Python IR wrapper + DSL wrapper

**文件**：`python/pypto/ir/op/prefetch_ops.py`（追加）

```python
def host_async(src: Expr, span: Span | None = None) -> Call:
    """Issue an orchestration-level GM->L2 prefetch for a flat 1D tensor.

    The prefetch is ordered before subsequently submitted tasks. Returns
    nothing; the runtime owns the issuance policy.
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("prefetch.host_async", [src], {}, actual_span)
```

**文件**：`python/pypto/language/op/prefetch_ops.py`（追加）

```python
def host_async(src: Tensor) -> None:
    """Prefetch a flat contiguous 1D GM tensor into L2 from the orchestration layer.

    A latency-hiding cache hint: the runtime issues the prefetch ahead of the
    tasks submitted after this statement, so their ``tile.load`` calls hit warm
    L2 lines. Changes no tensor values.

    Args:
        src: flat contiguous 1D GM tensor to prefetch.
    """
    _ir_prefetch.host_async(_unwrap(src))
```

**分发**：`ast_parser.py:6790-6793` 已有 `pl.prefetch.<op>` 三段式分发
（`_parse_prefetch_op` → `_dispatch_op`），DSL 模块新增 `host_async` 属性即可被解析，
无需改 parser。`__all__` 增加 `"host_async"`。

### 5.4 层 C1：Orchestration codegen

**文件**：`src/codegen/tensor_op_codegen.cpp`（追加）

```cpp
REGISTER_ORCHESTRATION_OP(prefetch_host_async, ("prefetch.host_async")) {
  // prefetch.host_async(tensor) -> rt_cmo_prefetch_async(tensor, bytes)
  //
  // Emit a call to the runtime's rt_cmo_prefetch_async(tensor, bytes). The
  // runtime owns the issuance policy — host-stream aclrtCmoAsync on onboard
  // a5, an equivalent AICPU-issued CMO on other targets, or a no-op on the
  // simulator — exactly the way tensor.read defers access policy to
  // get_tensor_data. Passing the TensorRef (not a raw address) lets the
  // runtime resolve the device address and producer dependency itself.
  INTERNAL_CHECK_SPAN(op->args_.size() == 1, op->span_)
      << "prefetch.host_async requires 1 argument";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "prefetch.host_async input must be a variable";

  auto input_type = AsTensorTypeLike(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(input_type, op->span_)
      << "Internal error: prefetch.host_async input must be TensorType, got "
      << op->args_[0]->GetType()->TypeName();

  std::string tensor_ref = codegen.GetExternalTensorName(input_name);

  // bytes = prod(shape) * sizeof(dtype)，发射为编译期/运行期标量表达式。
  // 维度表达式发射复用 EmitRuntimeTensorShapeDim（见 tensor_create 的模式）。
  std::ostringstream bytes;
  ...  // numel * dtype_size, 与 tensor.create 的 shape 发射同模式

  return "rt_cmo_prefetch_async(" + tensor_ref + ", " + bytes.str() + ");";
}
```

生成代码示例（Orchestration C++）：

```cpp
rt_cmo_prefetch_async(x, 524288);   // x 为 simpler::hbg::Tensor 的外部引用
TaskOutputTensors task_0_outs = rt_submit_aic_task(kernel_id, args);
```

### 5.5 simpler 运行时契约（关键开放依赖）

**新环境函数**（与 `rt_submit_task` / `get_tensor_data` 同域，声明于
`runtime/src/common/host_build_graph/types.h` 一族）：

```cpp
// Issue a GM->L2 CMO prefetch (cache hint) for `bytes` of `tensor`'s device
// buffer. Ordered before tasks submitted after this call. The tensor's
// producer must have completed (address materialized).
void rt_cmo_prefetch_async(const simpler::hbg::Tensor &tensor, uint64_t bytes);
```

实现变体（由 simpler 团队按平台落点选择，PyPTO 侧契约不变）：

| 变体 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| **V1（推荐先落）** | AICPU 把 prefetch 记录附加到随后的任务提交；host 侧 launch 路径在设备流上先调 `aclrtCmoAsync(..., ACL_RT_CMO_TYPE_PREFETCH, stream)` 再启动任务 | 忠实 SHMEM 路径 A（host API、流序免费）；复用已有 host↔AICPU 协调 | prefetch 与任务绑定关系由 runtime 内部维护 |
| V2（后续优化） | AICPU 直接向 STARS workspace 提交 CMO SQE（与 AIV 路径同机制） | 无 host 往返 | 需验证 AICPU 侧 DCCI/doorbell 语义，工作量大 |
| sim | no-op（保持"cache hint 不影响数值"契约） | — | — |

### 5.6 能力门控

**文件**：`include/pypto/backend/common/backend_handler.h`（追加虚方法，模式同
`l2-buffer-managed-design.md` §G1）

```cpp
/// Whether the backend supports orchestration-level host-issued CMO prefetch
/// (prefetch.host_async). Requires CANN >= 9.1.0 on Ascend950.
[[nodiscard]] virtual bool SupportsHostCmoPrefetch() const { return false; }
```

- `src/backend/950/backend_950_handler.cpp`：override 返回 `true`（CANN ≥ 9.1.0，与
  SDMA workspace 的版本约束一致）。
- 拒绝点：codegen 前置校验（复用 `src/codegen/codegen_preconditions.cpp` 的既有检查
  框架）——"Orchestration 函数含 `prefetch.host_async` 但 `!SupportsHostCmoPrefetch()`"
  时报清晰错误；sim 后端映射为 no-op 而非报错。

### 5.7 跨层同步修改清单

按 `.claude/rules/cross-layer-sync.md` 组织（本 op 无新 C++ 类/绑定面——op 注册经
`create_op_call` 字符串分发，无 nanobind 新方法，故无 .pyi 改动）：

| # | 文件 | 修改 |
|---|------|------|
| 1 | `src/ir/op/prefetch/prefetch_async.cpp` | 追加 `REGISTER_OP("prefetch.host_async")` |
| 2 | `python/pypto/ir/op/prefetch_ops.py` | 追加 `host_async` IR wrapper |
| 3 | `python/pypto/language/op/prefetch_ops.py` | 追加 `host_async` DSL wrapper + `__all__` |
| 4 | `src/codegen/tensor_op_codegen.cpp` | 追加 `REGISTER_ORCHESTRATION_OP(prefetch_host_async, ...)` |
| 5 | `include/pypto/backend/common/backend_handler.h` | 追加 `SupportsHostCmoPrefetch()` |
| 6 | `src/backend/950/backend_950_handler.h/.cpp` | override `true` |
| 7 | `src/codegen/codegen_preconditions.cpp` | 不支持后端 + 出现 host_async → 报错 |
| 8 | `tests/ut/ir/operators/test_prefetch_ops.py`（或同目录新文件） | IR 构建/约束测试 |
| 9 | `tests/ut/codegen/test_orchestration_codegen.py` | 生成文本含 `rt_cmo_prefetch_async` |
| 10 | `tests/st/runtime/ops/test_prefetch_host_async.py` | 新建 ST（非破坏性 + a5） |
| 11 | `docs/en/dev/ir/05-operators.md` + `docs/zh/dev/ir/05-operators.md` | op 表条目（en 为基准） |
| 12 | simpler（下游仓） | `rt_cmo_prefetch_async` 契约实现（V1/sim） |

### 5.8 测试策略

| 层 | 用例 | 断言 |
|----|------|------|
| IR UT | `test_host_async_build` | `Call` op 名为 `prefetch.host_async`，statement 无返回值 |
| IR UT | `test_host_async_rejects_2d` | 非 flat-1D 源报错（复用 async_prefetch 的既有测试模式） |
| IR UT | `test_host_async_rejects_incore` | InCore 函数中出现该 TensorOp 被归属检查拒绝 |
| Codegen UT | `test_host_async_orch_codegen` | 生成的 Orchestration C++ 含 `rt_cmo_prefetch_async(x, <bytes>)`，且 bytes = numel×dtype_size |
| Codegen UT | `test_host_async_unsupported_backend` | 910B handler（false）时报错信息明确 |
| ST（a5） | `test_host_async_does_not_perturb_data` | 预取 + 拷贝 golden 对比（与 #2089 ST 同范式）+ 等待完成不挂死 |

### 5.9 文档

- `docs/en/dev/ir/05-operators.md`（en 基准）+ `docs/zh/dev/ir/05-operators.md` 同步：
  prefetch op 家族表增加 `prefetch.host_async` 行。
- 用户文档随 ST 落地后补充 API 参考（`docs/en/user/api/` 对应页面）。

---

## 6. 可选增强（P2，需下游协同，暂不实施）

| 增强 | 内容 | 依赖 | 优先级 |
|------|------|------|--------|
| 显式 channel group / queue_num | `pl.prefetch.make_context(channel_group=None, queue_num=1)` 透传 pto-isa `BuildSdmaSession` 参数 | PTOAS `pto.make_prefetch_async_context` 需加属性 + pto-isa 读取（两仓联动） | 低——默认 auto（`get_block_idx()`）已是 SHMEM 实测最优语义 |
| CMO WRITEBACK/INVALID/FLUSH | `CMO_TYPE` 透传 | A5 硬件仅支持 PREFETCH（SHMEM 亦然） | 搁置 |

---

## 7. 实施阶段与依赖

```
阶段 0（前置，进行中）：A5 离线验证
  pto-isa ebb1d62e 离线验证通过 → 上游 PR → pypto bump pto_isa.pin
  （simpler patch + ST 平台扩展随同落地；路径 B/C 在 A5 端到端可用）
        │
阶段 1：prefetch.host_async PyPTO 侧全层（IR/DSL/codegen/门控/UT/文档）
  不阻塞于 simpler 新 API：codegen 生成 rt_cmo_prefetch_async 调用文本即可交付；
  sim 环境 no-op 使 ST 契约先行成立
        │
阶段 2：simpler rt_cmo_prefetch_async（V1：host 流序注入）+ a5 ST
        │
阶段 3（可选）：channel_group/queue_num 透传（PTOAS + pto-isa 联动）
```

每步验证标准（对应 `.claude/skills/add-op/SKILL.md` 检查清单）：

1. 阶段 1：`python -m pytest tests/ut/ir/operators/ -v -k host_async` 通过；
   `tests/ut/codegen/` codegen UT 通过；`pre-commit run --all-files` 通过。
2. 阶段 2：a5 ST 非破坏性断言通过；910B 报错路径覆盖。
3. 文档：en/zh 两个 `05-operators.md` 同步更新。

---

## 8. 备选方案与权衡

### 8.1 路径 A 的 PyPTO 侧形态

| 方案 | 描述 | 结论 |
|------|------|------|
| **A1（选定）** | orchestration op → 生成 C++ 调 `rt_cmo_prefetch_async` | 语义进 IR；运行时拥有策略（tensor.read 先例）；用户可在编排逻辑中自由定位预取点 |
| A2 | artifact 元数据 + host launch 前注入（类似 enable_sdma 握手） | 不需要 simpler 新环境函数，但"预取与哪个任务绑定"变成隐式元数据，跨 pass/序列化复杂，表达力弱（无法在循环内按迭代发起） |
| A3 | Python 运行时 API（`compiled.prefetch(x)`） | 破坏"语义承载在 IR"原则，与程序内任务序无法精确对齐；否决 |

### 8.2 simpler 侧落地变体

见 §5.5 表（V1 host 流序注入 vs V2 AICPU 直驱）。选择依据：`aclrtCmoAsync` 是
host API（AICore/AICPU 上不存在同名能力），V1 直接复用真实 API 与 SHMEM 实测路径；
V2 留作后续优化。

### 8.3 为什么不改用 `l2-buffer-a5-cmo-path.md` 的 wrapper 注入

该文档 §3.2 在"PTOAS 尚无 prefetch op"的约束下选择 wrapper 注入作为测试期最优。
此后 #2089 已把正式 IR op 路线（该文档 §9.3 的"阶段 2"）落地 main，前端语义、
pass 可见性、PTOAS 校验全部齐备，wrapper 注入方案自然作废。

---

## 9. 风险与开放问题

| # | 风险/问题 | 影响 | 缓解 |
|---|-----------|------|------|
| 1 | `rt_cmo_prefetch_async` 的 AICPU→host 落地机制（V1）需 simpler 团队确认 | 阶段 2 进度 | 契约先行（§5.5 签名冻结），sim no-op 保证阶段 1 独立交付 |
| 2 | CANN 9.1.0 版本约束（`aclrtCmoAsync` 与 SDMA workspace 同一版本门槛） | 现场环境 | 与 A5 使能三件套共用版本矩阵与自检（`a5-sdma-prefetch-minimal-guide.md` §2） |
| 3 | P2 增强需 PTOAS/pto-isa 两仓联动 | 透传能力延后 | 默认 auto 语义已是推荐值，增强非阻塞 |
| 4 | pto-isa pin bump 依赖上游 PR 合并节奏 | A5 端到端时间线 | 已有离线验证包推进中（fork `offline-verify-transfer`） |

---

## 10. 修改文件清单汇总

### 阶段 1（PyPTO 仓，本文档核心交付）

| 文件 | 类型 | 内容 |
|------|------|------|
| `src/ir/op/prefetch/prefetch_async.cpp` | 修改 | `prefetch.host_async` REGISTER_OP |
| `python/pypto/ir/op/prefetch_ops.py` | 修改 | `host_async` IR wrapper |
| `python/pypto/language/op/prefetch_ops.py` | 修改 | `host_async` DSL wrapper |
| `src/codegen/tensor_op_codegen.cpp` | 修改 | REGISTER_ORCHESTRATION_OP |
| `include/pypto/backend/common/backend_handler.h` | 修改 | `SupportsHostCmoPrefetch()` |
| `src/backend/950/backend_950_handler.h/.cpp` | 修改 | override |
| `src/codegen/codegen_preconditions.cpp` | 修改 | 后端门控检查 |
| `tests/ut/ir/operators/`、`tests/ut/codegen/` | 新增/扩展 | §5.8 UT |
| `tests/st/runtime/ops/test_prefetch_host_async.py` | 新建 | ST |
| `docs/en/dev/ir/05-operators.md`、`docs/zh/dev/ir/05-operators.md` | 修改 | op 表 |

### 阶段 2（simpler 仓，契约）

| 文件 | 内容 |
|------|------|
| `runtime/src/common/host_build_graph/types.h`（及实现文件） | `rt_cmo_prefetch_async` 声明 + per-arch 实现（a5 V1 / sim no-op） |

### 阶段 3（可选，PTOAS + pto-isa 联动）

| 仓 | 内容 |
|----|------|
| PTOAS | `pto.make_prefetch_async_context` 属性扩展 |
| pto-isa | `BuildSdmaSession` 参数从 context 读取 |
