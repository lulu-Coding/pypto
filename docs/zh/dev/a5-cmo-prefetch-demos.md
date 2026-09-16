# A5 CMO 预取最小 Demo 程序集（三条链路）

> **状态**：可编译示例（demo）
> **日期**：2026-09-16
> **目标**：为三条 GM→L2 预取链路各给出一个**最小可编译 C++ 程序**，完整展现接口调用过程。
> 全部 demo **不依赖 libshmem.so，也不依赖 Ascend C API（不包含 kernel_operator.h）**——
> host 侧只使用 ACL 公开 API + dlsym 符号 + AICPU 算子；设备侧只使用 ccec/bisheng
> **编译器原生层**（地址空间修饰、启动扩展、隐式内置函数与平台宏，清单见 §0.1）。
> **环境**：Ascend950（`__NPU_ARCH__=3510`）、CANN ≥ 9.1.0（toolkit + cann-950-ops-legacy 同装）、
> ccec（`--cce-aicore-arch=dav-c310`）或 bisheng（`--npu-arch=dav-3510`）
> **关联文档**：`a5-cmo-prefetch-shmem-free-minimal-impl.md`（接口清单 + 时序图/流程图）；
> `a5-shmem-prefetch-extraction.md`（SHMEM 仓内提取，字段级出处）
> **事实基准**：SQE 字段与初始化 = SHMEM PR #459 字节级对照 + pto-isa
> `fix/a5-stars-v2-sqe-init`（全字段显式初始化修复）；host 供给 = pto-isa
> `SdmaWorkspaceManager`（已验证编译）；quiet = pto-isa postId 机制；设备侧原生内置函数
> 惯用法 = pto-isa `hns_1825_backend.hpp`（A5 后端，**零 CANN 头编译**）+ pto-isa
> `include/pto/common/debug.h`（零 include 裸用 `trap()`/`cce::printf`）+ SHMEM
> `examples/cmo`（门控/启动惯例：950 每 block 1 AIC + 2 AIV，`get_block_idx()`=全局 AIV 序号）

---

## 0. 总览

| 文件 | 链路 | 角色 |
|---|---|---|
| `cmo_device.hpp` | B/C 公共 | 设备侧直驱：64B SQE 结构体 + `cmo_prefetch_nbi` / `cmo_quiet` 内联函数（**仅编译器原生内置函数，无 Ascend C**） |
| `sdma_provision.hpp` | B/C 公共 | host 供给链五步（dlsym → STARS 流 → workspace → H2D → AICPU StarsQuery） |
| `demo_a.cpp` | **A：host 流序** | 单文件，无供给链：`aclrtCmoAsync` 一次调用 |
| `demo_b.cpp` | **B：设备 QP0** | 供给 1 通道；单 AIV 内填 SQE + 鸣铃 + quiet |
| `demo_c.cpp` | **C：设备多 QP** | 供给 4 通道；每 AIV 用 `qp_idx = get_block_idx()` 独立通道并发预取 |

三条链路的接口调用要点（对应上一份文档的时序图 2/3）：

```
A:  aclrtCmoAsync(buf, bytes, ACL_RT_CMO_TYPE_PREFETCH, stream)   ← 全部
B:  [供给链5步] → kernel(AIV0): cmo_prefetch_nbi(ws,src,bytes,0) + cmo_quiet(ws,0)
C:  [供给链5步] → kernel(每AIV): cmo_prefetch_nbi(ws,my_slice,bytes,qp_idx) + cmo_quiet(ws,qp_idx)
                                qp_idx = get_block_idx()
```

### 0.1 设备侧依赖面 —— 编译器原生层清单（零 CANN 头）

B/C 链路的设备代码**不包含任何 CANN 头文件**（仅 `<cstdint>`），全部依赖由 ccec/bisheng
设备编译**隐式提供**（语言扩展 + 内置函数 + 平台宏），**不属于 Ascend C API 库**
（无需 `kernel_operator.h`）：

| 原生接口 | 作用 | 出处 / 已验证先例 |
|---|---|---|
| `__gm__` / `__ubuf__` | GM / UB 地址空间修饰 | 语言扩展 |
| `__global__ __aicore__` + `GM_ADDR` + `<<<blocks, nullptr, stream>>>` | kernel 入口与启动 | 语言扩展 |
| `set_flag(p0, p1, id)` / `wait_flag(p0, p1, id)` | 管线事件同步（S↔MTE3 等） | SHMEM 测试内核、pto-isa 内核均裸用 |
| `copy_ubuf_to_gm_align_v2(gm, ub, 0, 1, size, 0, 0, 0)` | MTE3 UB→GM 拷贝（4B/64B） | pto-isa `hns_1825_backend.hpp`（A5，零 CANN 头） |
| `dcci(ptr, SINGLE_CACHE_LINE)` | cache line 清理并失效 | pto-isa A5 内核（`ready_queue.hpp` / `moe_*` 等） |
| `ld_dev(ptr, 0)` / `st_dev(v, ptr, 0)` | 旁路标量 L1 的 GM 读/写 | pto-isa `hns_1825_backend.hpp`（A5，轮询 NIC 更新的队列索引） |
| `get_block_idx()` | 全局 AIV 编号（QP 定位） | pto-isa 内核（`get_block_idx() % mIter` 等） |
| `ASCEND_IS_NOT_AIV` / `ASCEND_IS_AIV` | 混合启动门控（AIC 直接返回） | SHMEM `examples/cmo`、pto-isa A5 内核 |
| `PIPE_S` / `PIPE_MTE2` / `PIPE_MTE3` / `PIPE_ALL`、`EVENT_ID0` | 管线与事件常量 | 同上 |
| `trap()` / `cce::printf` | 超时 abort / 设备侧打印 | pto-isa `include/pto/common/debug.h`（零 include） |

> 注 1：`AscendC::GetSystemCycle()` **不在**原生层（由 `kernel_operator_sys_var_intf.h` 声明，
> 属 API 层）——demo 的 quiet 超时因此改用轮询次数预算。
> 注 2：`ASCEND_IS_NOT_AIV` 为平台宏（无命名空间、非 API 对象），SHMEM 官方 CMO 示例与
> pto-isa A5 内核均裸用；950 上每个启动 block 为 1 AIC + 2 AIV 混合核组，AIC 不得触碰
> STARS 队列，故该门控是语义必需。

---

## 1. `cmo_device.hpp` —— 设备侧直驱（B/C 公共，仅编译器原生内置函数）

```cpp
// cmo_device.hpp — A5 (Ascend950) STARS v2 CMO prefetch. SHMEM-free AND
// Ascend-C-free: no libshmem, no kernel_operator.h — the device side uses only
// the compiler-native layer that ccec/bisheng provide implicitly for device
// compilation (inventory in §0.1):
//   language extensions : __gm__/__ubuf__, __global__/__aicore__, GM_ADDR, <<<>>>
//   implicit builtins   : set_flag / wait_flag, copy_ubuf_to_gm_align_v2,
//                         dcci, ld_dev / st_dev, get_block_idx, trap, cce::printf
//   platform macros     : ASCEND_IS_NOT_AIV, PIPE_S/PIPE_MTE3/PIPE_ALL,
//                         EVENT_ID0, SINGLE_CACHE_LINE
// Field layout mirrors SHMEM PR #459 (byte-verified); full-field SQE
// initialization mirrors pto-isa fix/a5-stars-v2-sqe-init. The raw-builtin
// idioms mirror pto-isa hns_1825_backend.hpp (A5 backend that compiles with
// no CANN include) and include/pto/common/debug.h (bare trap()/cce::printf).
#ifndef CMO_DEMO_DEVICE_HPP
#define CMO_DEMO_DEVICE_HPP

#include <cstdint>

#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ != 3510)
#error "This demo targets Ascend950 (__NPU_ARCH__=3510, --cce-aicore-arch=dav-c310)"
#endif

namespace cmo_demo {

// ---- constants (STARS v2 / A5) ----
constexpr uint8_t  kSqeTypeSdma        = 11;    // RT_STARS_SQE_TYPE_SDMA
constexpr uint8_t  kCmoPrefetchOpcode  = 6;     // == ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH
constexpr uint8_t  kKernelCredit       = 254;   // K_CREDIT_TIME_DEFAULT (STARS v2)
constexpr uint32_t kDoorbellOffset     = 0x0;   // STARS v2 (A5); v1/A2A3 uses 0x8
constexpr uint32_t kSqeBytes           = 64;
constexpr uint32_t kUbScratchOffset    = 1024;  // 64B-aligned UB scratch, >= 64B
constexpr uint32_t kFlagRegionOffset   = 8192;  // any 64B-aligned free area in the workspace
constexpr uint32_t kQuietPollLimit     = 100000000U;  // poll budget (~tens of seconds); SHMEM uses a 60 s cycle deadline

// ---- workspace structures (SHMEM-compatible layout) ----
struct StarsChannelFlagInfo {   // workspace + 0x0, 64B (filled by the AICPU op)
    uint32_t flag;
    uint32_t totalQueueNum;
    uint8_t  reserved[56];
};

struct StarsChannelInfo {       // workspace + 0x40 + i * 64B (filled by the AICPU op)
    uint32_t sq_head;           // +0
    uint32_t sq_tail;           // +4   software mirror of the tail
    uint64_t sq_base;           // +8   SQ ring GM base
    uint64_t sq_reg_base;       // +16  doorbell register GM base
    uint32_t sq_depth;          // +24
    uint32_t sq_id;             // +28
    uint32_t cq_id;             // +32
    uint32_t logic_cq_id;       // +36
    uint64_t cqe_addr;          // +40
    uint32_t report_cqe_num;    // +48
    uint32_t stream_id;         // +52
    uint32_t dev_id;            // +56
    uint8_t  reserved[4];       // +60
};

// 64B STARS v2 SQE. One layout serves both the CMO prefetch (opcode=6, dst=0)
// and the 8B quiet flag copy (opcode=0, dst=flag record).
struct StarsV2Sqe {
    // bytes 0..7: header
    uint8_t  type : 6;          // = 11 (SDMA)
    uint8_t  l1_lock : 1;       // = 0
    uint8_t  l1_unlock : 1;     // = 0
    uint8_t  ie : 1;            // = 0
    uint8_t  pre_p : 1;         // = 0
    uint8_t  post_p : 1;        // = 0
    uint8_t  wr_cqe : 1;        // = 1
    uint8_t  ptr_mode : 1;      // = 0
    uint8_t  rtt_mode : 1;      // = 0
    uint8_t  head_update : 1;   // = 0
    uint8_t  reserved0 : 1;     // = 0
    uint16_t num_blocks;        // = 0
    uint16_t rt_streamid;       // = channel_info->stream_id
    uint16_t task_id;           // = sq_tail - sq_head
    // bytes 8..15
    uint32_t res1;              // = 0
    uint16_t res2;              // = 0
    uint8_t  kernel_credit;     // = 254
    uint8_t  res3;              // = 0
    // bytes 16..19
    uint32_t opcode : 8;        // 6 = CMO PREFETCH, 0 = plain SDMA copy
    uint32_t sssv : 1;          // = 1
    uint32_t dssv : 1;          // = 1
    uint32_t sns : 1;           // = 1
    uint32_t dns : 1;           // = 1
    uint32_t sro : 1;           // = 0
    uint32_t dro : 1;           // = 0
    uint32_t stride : 2;        // = 0
    uint32_t ie2 : 1;           // = 0
    uint32_t comp_en : 1;       // = 0
    uint32_t res4 : 14;         // = 0
    // bytes 20..23
    uint16_t sqe_id;            // = 0
    uint8_t  mpam_partid;       // = 0
    uint8_t  mpamns : 1;        // = 0
    uint8_t  pmg : 2;           // = 0
    uint8_t  qos : 4;           // = 6 (HCCL QoS, matches SHMEM)
    uint8_t  res5 : 1;          // = 0
    // bytes 24..31
    uint16_t src_streamid;      // = 0
    uint16_t src_sub_streamid;  // = 0
    uint16_t dst_streamid;      // = 0
    uint16_t dst_sub_streamid;  // = 0
    // bytes 32..47
    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;
    // bytes 48..63
    uint32_t length;            // bytes to transfer
    uint32_t src_offset_low;    // = 0
    uint32_t dst_offset_low;    // = 0
    uint16_t src_offset_high;   // = 0
    uint16_t dst_offset_high;   // = 0
};
static_assert(sizeof(StarsV2Sqe) == 64, "STARS v2 SQE must be 64B");

// ---- workspace addressing ----
static __aicore__ inline __gm__ StarsChannelInfo* channel_at(__gm__ uint8_t* ws, uint32_t qp_idx) {
    return reinterpret_cast<__gm__ StarsChannelInfo*>(ws + sizeof(StarsChannelFlagInfo)) + qp_idx;
}

// quiet flag slots: one 64B slot per channel — [send u64 @+0][done u64 @+32]
static __aicore__ inline __gm__ uint8_t* flag_slot(__gm__ uint8_t* ws, uint32_t qp_idx) {
    return ws + kFlagRegionOffset + 64ull * qp_idx;
}

// ---- UB scratch (64B, 64B-aligned, at UB offset 1024 — the same offset
//      SHMEM's cmo example uses for its tmp buffer) ----
static __aicore__ inline __ubuf__ uint8_t* ub_scratch() {
    return reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(kUbScratchOffset));
}

// ---- 4B GM store through MTE3 (raw form of SHMEM aclshmemi_set_value /
//      pto-isa WriteUbToGmWithSync): the S->MTE3 pair orders the scalar UB
//      write before the copy, the MTE3->S pair orders the copy before any
//      later scalar work. MTE3 writes bypass the AIV data cache, so the value
//      lands in L2 for the SDMA engine / doorbell register. ----
static __aicore__ inline void store_u32_gm(__gm__ uint32_t* dst, uint32_t v) {
    __ubuf__ uint32_t* tmp = reinterpret_cast<__ubuf__ uint32_t*>(ub_scratch());
    *tmp = v;
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm_align_v2(dst, tmp, 0, 1, sizeof(uint32_t), 0, 0, 0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

// ---- cache-bypass scalar GM load (pto-isa hns_1825 ReadU32Gm idiom, used
//      there on A5 to poll queue indices written by the NIC): ld_dev skips
//      the AIV scalar L1, so read paths need no DCCI at all. ----
static __aicore__ inline uint32_t load_u32_gm(__gm__ uint32_t* src) {
    return ld_dev(src, 0);
}

// ---- DCCI: clean & invalidate the cache lines covering [addr, addr + bytes).
//      Raw form dcci(ptr, SINGLE_CACHE_LINE) as in pto-isa A5 kernels. A
//      single line covers the whole 64B channel_info. ----
static __aicore__ inline void dcci_range(__gm__ void* addr, uint32_t bytes) {
    __gm__ uint8_t* p = reinterpret_cast<__gm__ uint8_t*>(addr);
    for (uint32_t off = 0; off < bytes; off += 64) {
        dcci(reinterpret_cast<__gm__ void*>(p + off), SINGLE_CACHE_LINE);
    }
}

// ---- submit one 64B SQE on channel qp_idx and ring the doorbell ----
// opcode 6 + dst nullptr  -> CMO prefetch into L2 (fire and forget)
// opcode 0 + dst          -> plain 8B SDMA copy (used by cmo_quiet)
static __aicore__ inline void submit_sqe(
    __gm__ uint8_t* ws, uint32_t qp_idx, __gm__ void* src, __gm__ void* dst, uint32_t bytes,
    uint32_t opcode)
{
    __gm__ StarsChannelInfo* ci = channel_at(ws, qp_idx);

    // 1. fresh sq_tail / sq_head (ld_dev bypasses the scalar L1 — no DCCI needed)
    const uint32_t sq_tail = load_u32_gm(reinterpret_cast<__gm__ uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(ci) + 4));
    const uint32_t sq_head = load_u32_gm(reinterpret_cast<__gm__ uint32_t*>(ci));

    // 2. fully initialize the SQE with plain scalar GM stores (the same way
    //    SHMEM aclshmemi_fill_stars_v2_cmo_sqe writes it) — SQ ring memory is
    //    unspecified, every hardware-parsed field must be written
    //    (fix/a5-stars-v2-sqe-init)
    __gm__ StarsV2Sqe* sqe =
        reinterpret_cast<__gm__ StarsV2Sqe*>(ci->sq_base) + (sq_tail % ci->sq_depth);
    *sqe = StarsV2Sqe{};
    sqe->type          = kSqeTypeSdma;
    sqe->wr_cqe        = 1;
    sqe->num_blocks    = 0;
    sqe->rt_streamid   = static_cast<uint16_t>(ci->stream_id);
    sqe->task_id       = static_cast<uint16_t>(sq_tail - sq_head);
    sqe->kernel_credit = kKernelCredit;
    sqe->opcode        = opcode;
    sqe->sssv = 1; sqe->dssv = 1; sqe->sns = 1; sqe->dns = 1;
    sqe->qos           = 6;   // HCCL QoS
    const uint64_t s = reinterpret_cast<uint64_t>(src);
    const uint64_t d = reinterpret_cast<uint64_t>(dst);
    sqe->src_addr_low  = static_cast<uint32_t>(s & 0xFFFFFFFFull);
    sqe->src_addr_high = static_cast<uint32_t>(s >> 32);
    sqe->dst_addr_low  = static_cast<uint32_t>(d & 0xFFFFFFFFull);
    sqe->dst_addr_high = static_cast<uint32_t>(d >> 32);
    sqe->length        = bytes;

    // 3. flush the SQE line so STARS hardware sees it
    dcci_range(sqe, kSqeBytes);

    // 4. advance the tail; ring the doorbell register and mirror the tail
    const uint32_t next = (sq_tail + 1) % ci->sq_depth;
    store_u32_gm(reinterpret_cast<__gm__ uint32_t*>(ci->sq_reg_base + kDoorbellOffset), next);
    store_u32_gm(reinterpret_cast<__gm__ uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(ci) + 4), next);
}

// ---- CMO prefetch (non-blocking; pair with cmo_quiet on the same channel) ----
static __aicore__ inline void cmo_prefetch_nbi(
    __gm__ uint8_t* ws, __gm__ void* src, uint32_t bytes, uint32_t qp_idx)
{
    submit_sqe(ws, qp_idx, src, nullptr, bytes, kCmoPrefetchOpcode);
}

// ---- quiet: wait until all previously submitted SQEs on qp_idx have drained.
//      Two-slot postId scheme (pto-isa): arm send=1, submit an 8B SDMA copy
//      send -> done queued AFTER the data SQEs (its landing proves the queue
//      drained), poll done through the cache-bypass load, then reset both
//      slots. SHMEM arms a 4B value against the 8B copy — the upper word
//      stays 0 forever, so polling the low word is sufficient. ----
static __aicore__ inline void cmo_quiet(__gm__ uint8_t* ws, uint32_t qp_idx)
{
    __gm__ uint8_t* slot = flag_slot(ws, qp_idx);
    __gm__ uint32_t* send = reinterpret_cast<__gm__ uint32_t*>(slot);        // low word of the u64 slot
    __gm__ uint32_t* done = reinterpret_cast<__gm__ uint32_t*>(slot + 32);   // low word of the u64 slot

    // 1. arm the flag (through MTE3 so it lands in L2 for the SDMA engine)
    store_u32_gm(send, 1u);

    // 2. queue the 8B flag SQE after the data SQEs
    submit_sqe(ws, qp_idx, slot, slot + 32, 8, /*opcode=*/0);

    // 3. poll the done slot until non-zero. A poll-count budget replaces
    //    SHMEM's 60 s cycle deadline because GetSystemCycle() is an AscendC
    //    API (kernel_operator_sys_var_intf.h), not a native builtin.
    uint32_t v = 0;
    for (uint32_t i = 0; i < kQuietPollLimit && v == 0; ++i) {
        v = load_u32_gm(done);
    }
    if (v == 0) {
        cce::printf("cmo_quiet: timeout on channel %u\n", qp_idx);  // native builtin (pto-isa debug.h)
        trap();                                                     // SHMEM aclshmemi_kernel_abort ends in trap()
    }

    // 4. reset both slots for the next round
    store_u32_gm(send, 0u);
    store_u32_gm(done, 0u);
}

}  // namespace cmo_demo

#endif  // CMO_DEMO_DEVICE_HPP
```

---

## 2. `sdma_provision.hpp` —— host 供给链（B/C 公共，五步）

```cpp
// sdma_provision.hpp — SHMEM-free host provisioning for the CMO demo.
// Condensed from pto-isa SdmaWorkspaceManager (hardware-verified compile),
// with the channel count parameterized. Five steps (see the sequence diagram
// in a5-cmo-prefetch-shmem-free-minimal-impl.md §3):
//   0) dlsym libruntime/libopapi   1) N STARS streams   2) 16KB workspace
//   3) stream table H2D            4) AICPU StarsQuery op programs channels
#ifndef CMO_DEMO_SDMA_PROVISION_HPP
#define CMO_DEMO_SDMA_PROVISION_HPP

#ifndef __CCE_KT_TEST__  // host-only header: hidden from the device pass

#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <vector>

#include "acl/acl.h"

#ifndef ACL_STREAM_DEVICE_USE_ONLY
#define ACL_STREAM_DEVICE_USE_ONLY 0x00000020U
#endif
#ifndef ACL_STREAM_FAST_LAUNCH
#define ACL_STREAM_FAST_LAUNCH 0x00000004U
#endif
#ifndef ACL_STREAM_FAST_SYNC
#define ACL_STREAM_FAST_SYNC 0x00000008U
#endif

// aclnn tensor API (libnnopbase) — forward-declared, same as pto-isa.
struct aclTensor;
struct aclOpExecutor;
extern "C" aclTensor* aclCreateTensor(
    const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t* stride,
    int64_t offset, aclFormat format, const int64_t* storageDims, uint64_t storageDimsNum,
    void* tensorData);
extern "C" int32_t aclDestroyTensor(const aclTensor* tensor);

namespace cmo_demo {

constexpr uint32_t kMaxChannels    = 48;             // pto-isa kSdmaMaxChannelGroups
constexpr uint32_t kWorkspaceBytes = 16U * 1024U;    // ctx region + quiet flag region

struct HostStreamInfo {   // 64B, one per STARS stream (device-visible layout)
    uint64_t stream_;
    uint64_t ctx_;
    int32_t  stream_id;
    uint32_t sq_id;
    uint32_t cq_id;
    uint32_t logic_cq_id;
    uint64_t cqe_addr;
    int32_t  dev_id;
    uint8_t  reserved[20];
};
static_assert(sizeof(HostStreamInfo) == 64);

struct SdmaOpResInfo {    // 64B, device-visible
    uint64_t size;            // stream count
    uint64_t streams_addr;    // GM address of HostStreamInfo[]
    uint64_t workspace_addr;  // GM address of the workspace
    uint8_t  reserved[40];
};
static_assert(sizeof(SdmaOpResInfo) == 64);

class SdmaProvisioner {
public:
    bool Init(uint32_t channel_num) {
        if (inited_ || channel_num == 0 || channel_num > kMaxChannels) return false;
        channel_num_ = channel_num;
        return LoadSymbols() &&          // step 0
               CreateStarsStreams() &&   // step 1
               MallocWorkspace() &&      // step 2
               CopyOpResToDevice() &&    // step 3
               LaunchStarsQuery();       // step 4
    }

    void Finalize() {   // release in reverse order
        if (opres_dev_)   { aclrtFree(opres_dev_);              opres_dev_ = nullptr; }
        if (streams_dev_) { aclrtFree(streams_dev_);            streams_dev_ = nullptr; }
        if (workspace_)   { aclrtFree(workspace_);              workspace_ = nullptr; }
        for (auto& s : streams_) {
            if (s.stream_) { aclrtDestroyStream(reinterpret_cast<aclrtStream>(s.stream_)); s.stream_ = 0; }
        }
        streams_.clear();
        if (opapi_handle_) { dlclose(opapi_handle_); opapi_handle_ = nullptr; }
        if (rt_handle_)    { dlclose(rt_handle_);    rt_handle_ = nullptr; }
    }

    void*    workspace() const { return workspace_; }
    uint32_t channel_num() const { return channel_num_; }

    // debug helper: D2H-read channel 0 and print what the AICPU op filled in
    void DumpChannel0() {
        uint8_t ctx[64 + 64];
        if (aclrtMemcpy(ctx, sizeof(ctx), workspace_, sizeof(ctx), ACL_MEMCPY_DEVICE_TO_HOST) != 0) return;
        auto* ci = reinterpret_cast<StarsChannelInfoView*>(ctx + 64);
        printf("[provision] ch0: sq_base=0x%llx sq_reg_base=0x%llx depth=%u stream_id=%u\n",
               (unsigned long long)ci->sq_base, (unsigned long long)ci->sq_reg_base,
               ci->sq_depth, ci->stream_id);
    }

private:
    struct StarsChannelInfoView {   // host-side view of the 64B channel info
        uint32_t sq_head, sq_tail;
        uint64_t sq_base, sq_reg_base;
        uint32_t sq_depth, sq_id, cq_id, logic_cq_id;
        uint64_t cqe_addr;
        uint32_t report_cqe_num, stream_id, dev_id;
        uint8_t  reserved[4];
    };

    // step 0: dynamic symbols
    bool LoadSymbols() {
        rt_handle_ = dlopen("libruntime.so", RTLD_NOW);
        opapi_handle_ = dlopen("libopapi.so", RTLD_NOW);
        if (!rt_handle_ || !opapi_handle_) {
            printf("[provision] dlopen failed — CANN >= 9.1.0 required\n");
            return false;
        }
        rt_stream_get_sqid_ = reinterpret_cast<int32_t (*)(const void*, uint32_t*)>(
            dlsym(rt_handle_, "rtStreamGetSqid"));
        rt_stream_get_cqid_ = reinterpret_cast<int32_t (*)(const void*, uint32_t*, uint32_t*)>(
            dlsym(rt_handle_, "rtStreamGetCqid"));
        rt_get_device_info_ = reinterpret_cast<int32_t (*)(uint32_t, int32_t, int32_t, int64_t*)>(
            dlsym(rt_handle_, "rtGetDeviceInfo"));
        aclnn_get_ws_size_ = reinterpret_cast<int32_t (*)(aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**)>(
            dlsym(opapi_handle_, "aclnnShmemSdmaStarsQueryGetWorkspaceSize"));
        aclnn_exec_ = reinterpret_cast<int32_t (*)(void*, uint64_t, aclOpExecutor*, aclrtStream)>(
            dlsym(opapi_handle_, "aclnnShmemSdmaStarsQuery"));
        if (!rt_stream_get_sqid_ || !rt_stream_get_cqid_ || !rt_get_device_info_ ||
            !aclnn_get_ws_size_ || !aclnn_exec_) {
            printf("[provision] dlsym failed — is cann-950-ops-legacy 9.1.0 installed?\n");
            return false;
        }
        return true;
    }

    // step 1: one ACL_STREAM_DEVICE_USE_ONLY stream per channel
    bool CreateStarsStreams() {
        int32_t device_id = -1;
        if (aclrtGetDevice(&device_id) != 0) return false;
        int64_t die_id = -1;
        constexpr int32_t kInfoTypePhyDieId = 19;
        if (rt_get_device_info_(static_cast<uint32_t>(device_id), 0, kInfoTypePhyDieId, &die_id) != 0) {
            return false;
        }
        streams_.resize(channel_num_);
        for (uint32_t i = 0; i < channel_num_; ++i) {
            HostStreamInfo& info = streams_[i];
            void* stream = nullptr;
            if (aclrtCreateStreamWithConfig(reinterpret_cast<aclrtStream*>(&stream), 0,
                                            ACL_STREAM_DEVICE_USE_ONLY) != 0) {
                return false;
            }
            info.stream_ = reinterpret_cast<uint64_t>(stream);
            if (aclrtStreamGetId(reinterpret_cast<aclrtStream>(stream), &info.stream_id) != 0 ||
                rt_stream_get_sqid_(stream, &info.sq_id) != 0 ||
                rt_stream_get_cqid_(stream, &info.cq_id, &info.logic_cq_id) != 0) {
                return false;
            }
            void* ctx = nullptr;
            if (aclrtGetCurrentContext(&ctx) != 0) return false;
            info.ctx_    = reinterpret_cast<uint64_t>(ctx);
            info.dev_id  = static_cast<int32_t>(die_id);
        }
        printf("[provision] created %u STARS streams\n", channel_num_);
        return true;
    }

    // step 2: 16KB zeroed workspace
    bool MallocWorkspace() {
        if (aclrtMalloc(&workspace_, kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
            aclrtMemset(workspace_, kWorkspaceBytes, 0, kWorkspaceBytes) != 0) {
            return false;
        }
        return true;
    }

    // step 3: copy the stream table and the op-res descriptor to GM
    bool CopyOpResToDevice() {
        const size_t streams_bytes = streams_.size() * sizeof(HostStreamInfo);
        if (aclrtMalloc(&streams_dev_, streams_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
            aclrtMemcpy(streams_dev_, streams_bytes, streams_.data(), streams_bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
            return false;
        }
        SdmaOpResInfo op_res{};
        op_res.size            = streams_.size();
        op_res.streams_addr    = reinterpret_cast<uint64_t>(streams_dev_);
        op_res.workspace_addr  = reinterpret_cast<uint64_t>(workspace_);
        if (aclrtMalloc(&opres_dev_, sizeof(op_res), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
            aclrtMemcpy(opres_dev_, sizeof(op_res), &op_res, sizeof(op_res),
                        ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
            return false;
        }
        return true;
    }

    // step 4: AICPU StarsQuery op programs the STARS channels into the workspace.
    // It receives [streams_addr, workspace_addr] as one uint64 aclTensor pair.
    bool LaunchStarsQuery() {
        aclrtStream aicpu_stream = nullptr;
        if (aclrtCreateStreamWithConfig(&aicpu_stream, 0,
                                        ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC) != 0) {
            return false;
        }
        aclrtStreamAttrValue attr{};
        attr.failureMode = 1;
        aclrtSetStreamAttribute(aicpu_stream, ACL_STREAM_ATTR_FAILURE_MODE, &attr);

        const uint64_t in_data[2] = {reinterpret_cast<uint64_t>(streams_dev_),
                                     reinterpret_cast<uint64_t>(workspace_)};
        const uint64_t out_data[1] = {0};
        aclTensor* in  = MakeU64Tensor(in_data, 2);
        aclTensor* out = MakeU64Tensor(out_data, 1);
        if (!in || !out) return false;

        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        bool ok = aclnn_get_ws_size_(in, out, &ws_size, &executor) == 0;
        void* ws = nullptr;
        if (ok && ws_size > 0) {
            ok = aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST) == 0;
        }
        if (ok) {
            ok = aclnn_exec_(ws, ws_size, executor, aicpu_stream) == 0 &&
                 aclrtSynchronizeStream(aicpu_stream) == 0;
        }
        if (ws) aclrtFree(ws);
        (void)aclDestroyTensor(in);
        (void)aclDestroyTensor(out);
        aclrtDestroyStream(aicpu_stream);
        if (ok) printf("[provision] StarsQuery AICPU op completed\n");
        return ok;
    }

    // helper: uint64 aclTensor backed by a device copy of host data
    aclTensor* MakeU64Tensor(const uint64_t* host_data, uint64_t count) {
        const size_t bytes = count * sizeof(uint64_t);
        void* dev = nullptr;
        if (aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
            aclrtMemcpy(dev, bytes, host_data, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
            return nullptr;
        }
        const int64_t shape[1] = {static_cast<int64_t>(count)};
        const int64_t stride[1] = {1};
        aclTensor* t = aclCreateTensor(shape, 1, aclDataType::ACL_UINT64, stride, 0,
                                       aclFormat::ACL_FORMAT_ND, shape, 1, dev);
        if (!t) aclrtFree(dev);   // tensor owns dev on success (leak accepted in demo)
        return t;
    }

    uint32_t channel_num_ = 0;
    void* workspace_ = nullptr;
    void* streams_dev_ = nullptr;
    void* opres_dev_ = nullptr;
    std::vector<HostStreamInfo> streams_;
    void* rt_handle_ = nullptr;
    void* opapi_handle_ = nullptr;
    int32_t (*rt_stream_get_sqid_)(const void*, uint32_t*) = nullptr;
    int32_t (*rt_stream_get_cqid_)(const void*, uint32_t*, uint32_t*) = nullptr;
    int32_t (*rt_get_device_info_)(uint32_t, int32_t, int32_t, int64_t*) = nullptr;
    int32_t (*aclnn_get_ws_size_)(aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**) = nullptr;
    int32_t (*aclnn_exec_)(void*, uint64_t, aclOpExecutor*, aclrtStream) = nullptr;
};

}  // namespace cmo_demo

#endif  // __CCE_KT_TEST__
#endif  // CMO_DEMO_SDMA_PROVISION_HPP
```

---

## 3. Demo A —— 路径 A：host 流序预取（无供给链）

整个链路只有一个接口调用；同 stream 的后续 kernel 自动排在预取之后。

```cpp
// demo_a.cpp — Path A: host-issued, stream-ordered CMO prefetch.
// No provisioning at all: aclrtCmoAsync on a stream; anything submitted to the
// same stream afterwards is ordered behind the prefetch. CANN >= 9.1.0.
// Build: g++ -std=c++17 -I$ASCEND_HOME_PATH/include demo_a.cpp \
//            -L$ASCEND_HOME_PATH/lib64 -lascendcl -o demo_a
#include <cstdio>
#include "acl/acl.h"

#define CHECK(expr)                                                        \
    do {                                                                   \
        aclError _e = (expr);                                              \
        if (_e != ACL_SUCCESS) {                                           \
            printf("%s:%d acl err %d\n", __FILE__, __LINE__, (int)_e);     \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main() {
    const uint32_t bytes = 8u * 1024 * 1024;  // prefetch 8MB

    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream));

    void* buf = nullptr;
    CHECK(aclrtMalloc(&buf, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemset(buf, bytes, 0xAB, bytes));  // known pattern — must survive

    // ---- the entire path A ----
    CHECK(aclrtCmoAsync(buf, bytes, ACL_RT_CMO_TYPE_PREFETCH, stream));
    // a consumer kernel launched on `stream` after this point hits warm L2:
    //   my_kernel<<<..., stream>>>(buf, ...);
    //   CHECK(aclrtSynchronizeStream(stream));

    CHECK(aclrtSynchronizeStream(stream));

    // verify the cache hint is non-destructive
    unsigned char check[4096];
    CHECK(aclrtMemcpy(check, sizeof(check), buf, sizeof(check), ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t i = 0; i < sizeof(check); ++i) {
        if (check[i] != 0xAB) { printf("FAIL: data perturbed at %zu\n", i); return 1; }
    }
    printf("[demo_a] host CMO prefetch OK (8MB prefetched, data intact)\n");

    CHECK(aclrtFree(buf));
    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    return 0;
}
```

---

## 4. Demo B —— 路径 B：设备 QP0 单 AIV

```cpp
// demo_b.cpp — Path B: device-side prefetch on QP0 from a single AIV.
// host: provision 1 channel -> launch a 1-block kernel -> sync.
// device: AIV 0 fills one STARS v2 CMO SQE on channel 0, rings the doorbell,
//         then quiets (the stream sync alone does NOT cover doorbell-submitted
//         SQEs — that is why quiet exists).
#include "cmo_device.hpp"
#include "sdma_provision.hpp"
#include "acl/acl.h"
#include <cstdio>

#define CHECK(expr)                                                        \
    do {                                                                   \
        aclError _e = (expr);                                              \
        if (_e != ACL_SUCCESS) {                                           \
            printf("%s:%d acl err %d\n", __FILE__, __LINE__, (int)_e);     \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// ---------------- device kernel ----------------
__global__ __aicore__ void prefetch_qp0(GM_ADDR ws, GM_ADDR src, uint32_t bytes) {
    if (ASCEND_IS_NOT_AIV) { return; }     // mixed block: AIC lanes stay out (CMO is AIV-only)
    if (get_block_idx() != 0) { return; }  // QP0 interfaces: AIV 0 only
    __gm__ uint8_t* ws_gm = reinterpret_cast<__gm__ uint8_t*>(ws);
    cmo_demo::cmo_prefetch_nbi(ws_gm, reinterpret_cast<__gm__ void*>(src), bytes, /*qp_idx=*/0);
    cmo_demo::cmo_quiet(ws_gm, /*qp_idx=*/0);
}

// ---------------- host ----------------
int main() {
    const uint32_t bytes = 8u * 1024 * 1024;

    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream));

    // ---- provisioning, once per process (5 steps inside) ----
    cmo_demo::SdmaProvisioner prov;
    if (!prov.Init(/*channel_num=*/1)) {
        printf("[demo_b] provisioning failed (CANN >= 9.1.0 + 950-ops-legacy?)\n");
        return 1;
    }
    prov.DumpChannel0();  // sanity: sq_base/sq_reg_base/depth filled by the AICPU op

    void* buf = nullptr;
    CHECK(aclrtMalloc(&buf, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemset(buf, bytes, 0xAB, bytes));

    // ---- device-side prefetch on QP0: SQE fill + DCCI + doorbell + quiet ----
    prefetch_qp0<<<1, nullptr, stream>>>(reinterpret_cast<GM_ADDR>(prov.workspace()),
                                         reinterpret_cast<GM_ADDR>(buf), bytes);
    CHECK(aclrtSynchronizeStream(stream));

    unsigned char check[4096];
    CHECK(aclrtMemcpy(check, sizeof(check), buf, sizeof(check), ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t i = 0; i < sizeof(check); ++i) {
        if (check[i] != 0xAB) { printf("FAIL: data perturbed at %zu\n", i); return 1; }
    }
    printf("[demo_b] device QP0 prefetch + quiet OK (8MB, data intact)\n");

    prov.Finalize();
    CHECK(aclrtFree(buf));
    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    return 0;
}
```

---

## 5. Demo C —— 路径 C：设备多 AIV，每 AIV 独立 QP

```cpp
// demo_c.cpp — Path C: every AIV prefetches its own slice on its own QP.
// qp_idx = get_block_idx() — the global AIV index selects the channel, exactly
// the semantics SHMEM measured as optimal (AscendC::GetBlockIdx) and pto-isa
// implements as kAutoChannelGroupIdx -> get_block_idx().
#include "cmo_device.hpp"
#include "sdma_provision.hpp"
#include "acl/acl.h"
#include <cstdio>

#define CHECK(expr)                                                        \
    do {                                                                   \
        aclError _e = (expr);                                              \
        if (_e != ACL_SUCCESS) {                                           \
            printf("%s:%d acl err %d\n", __FILE__, __LINE__, (int)_e);     \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// ---------------- device kernel ----------------
__global__ __aicore__ void prefetch_per_aiv(
    GM_ADDR ws, GM_ADDR src, uint32_t bytes_per_aiv, uint32_t qp_num) {
    if (ASCEND_IS_NOT_AIV) { return; }
    const uint32_t qp_idx = get_block_idx();  // global AIV index == QP index
    if (qp_idx >= qp_num) { return; }
    __gm__ uint8_t* ws_gm = reinterpret_cast<__gm__ uint8_t*>(ws);
    __gm__ uint8_t* my_slice =
        reinterpret_cast<__gm__ uint8_t*>(src) + static_cast<uint64_t>(bytes_per_aiv) * qp_idx;
    cmo_demo::cmo_prefetch_nbi(ws_gm, my_slice, bytes_per_aiv, qp_idx);
    cmo_demo::cmo_quiet(ws_gm, qp_idx);
}

// ---------------- host ----------------
int main() {
    const uint32_t qp_num = 4;                    // 4 AIVs -> 4 channels
    const uint32_t bytes_per_aiv = 8u * 1024 * 1024;
    const uint64_t total = static_cast<uint64_t>(qp_num) * bytes_per_aiv;

    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream));

    // ---- provisioning: qp_num channels ----
    cmo_demo::SdmaProvisioner prov;
    if (!prov.Init(qp_num)) {
        printf("[demo_c] provisioning failed (CANN >= 9.1.0 + 950-ops-legacy?)\n");
        return 1;
    }
    prov.DumpChannel0();

    void* buf = nullptr;
    CHECK(aclrtMalloc(&buf, total, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemset(buf, total, 0xAB, total));

    // ---- per-AIV prefetch: 2 AIVs per block on 950 -> qp_num/2 blocks ----
    prefetch_per_aiv<<<(qp_num + 1) / 2, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(prov.workspace()), reinterpret_cast<GM_ADDR>(buf),
        bytes_per_aiv, qp_num);
    CHECK(aclrtSynchronizeStream(stream));

    unsigned char check[4096];
    CHECK(aclrtMemcpy(check, sizeof(check),
                      static_cast<char*>(buf) + total - sizeof(check), sizeof(check),
                      ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t i = 0; i < sizeof(check); ++i) {
        if (check[i] != 0xAB) { printf("FAIL: data perturbed at %zu\n", i); return 1; }
    }
    printf("[demo_c] per-AIV QP prefetch OK (%u AIVs x 8MB, data intact)\n", qp_num);

    prov.Finalize();
    CHECK(aclrtFree(buf));
    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    return 0;
}
```

---

## 6. 编译与运行

### 6.1 环境自检

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info                                        # 950 在列
which ccec || which bisheng                         # 编译器在 PATH
nm -D $ASCEND_HOME_PATH/lib64/libopapi.so | grep -i ShmemSdmaStarsQuery
#   ↑ 必须有输出，否则 CANN/ops 包版本不满足 9.1.0
```

### 6.2 编译

```bash
# Demo A：纯 host 程序，普通编译即可
g++ -std=c++17 -I$ASCEND_HOME_PATH/include demo_a.cpp \
    -L$ASCEND_HOME_PATH/lib64 -lascendcl -o demo_a

# Demo B/C：host+device 同文件，用 ccec（与 SHMEM 示例同族命令）
# 若用 bisheng：-xasc + --npu-arch=dav-3510 替换 -x cce + --cce-aicore-arch=dav-c310
COMMON_FLAGS="-O2 -x cce -std=c++17 --cce-aicore-arch=dav-c310 \
  -mllvm -cce-aicore-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false \
  -I. -I$ASCEND_HOME_PATH/include"

ccec $COMMON_FLAGS --cce-fatobj-link demo_b.cpp \
    -L$ASCEND_HOME_PATH/lib64 -lascendcl -lnnopbase -ldl -o demo_b
ccec $COMMON_FLAGS --cce-fatobj-link demo_c.cpp \
    -L$ASCEND_HOME_PATH/lib64 -lascendcl -lnnopbase -ldl -o demo_c
```

要点：
- `-cce-aicore-dcci-insert-for-scalar=false` 必须保留——设备侧自管 DCCI，编译器重复插入
  会破坏 SQE/tail 的一致性时序（SHMEM 构建同为该组合）。
- 设备代码**不包含任何 CANN 头**（`kernel_operator.h` 也不需要）：`set_flag`/`wait_flag`/
  `copy_ubuf_to_gm_align_v2`/`dcci`/`ld_dev`/`st_dev`/`get_block_idx`/`trap`/`cce::printf`
  与 `PIPE_*`/`EVENT_ID0`/`SINGLE_CACHE_LINE`/`ASCEND_IS_NOT_AIV` 均由 ccec/bisheng 设备编译
  隐式提供（pto-isa `hns_1825_backend.hpp` 与 `include/pto/common/debug.h` 即零 CANN 头裸用
  的先例，见 §0.1）。若现场编译器版本未隐式提供其中某个名字，补包含对应编译器平台头即可，
  demo 逻辑不变。

### 6.3 运行与预期输出

```bash
./demo_a
# [demo_a] host CMO prefetch OK (8MB prefetched, data intact)

./demo_b
# [provision] created 1 STARS streams
# [provision] StarsQuery AICPU op completed
# [provision] ch0: sq_base=0x... sq_reg_base=0x... depth=... stream_id=...
# [demo_b] device QP0 prefetch + quiet OK (8MB, data intact)

./demo_c
# [provision] created 4 STARS streams
# [provision] StarsQuery AICPU op completed
# [provision] ch0: sq_base=0x... sq_reg_base=0x... depth=... stream_id=...
# [demo_c] per-AIV QP prefetch OK (4 AIVs x 8MB, data intact)
```

**验证判据**：① 数据模式 `0xAB` 完好（预取是纯 cache hint，非破坏性）；② 程序正常退出
（quiet 未超时 = SQE 真正下到了硬件）；③ `[provision] ch0` 行的 `sq_base/sq_reg_base/depth`
非零（AICPU 算子确实编程了通道——若全零说明供给链没走通，kernel 内 doorbell 写会踩空）。

---

## 7. demo 相对参考实现的简化点（读代码前须知）

| 简化 | 参考实现的做法 | 影响 |
|---|---|---|
| 每次 submit 都 `ld_dev` 重读 GM tail/head | pto-isa 在 session 建立时读一次，之后 tail 保存在寄存器（`PersistSqTails` 回写） | demo 语义等价、性能略低，逻辑更直观 |
| quiet 用双槽 postId（send/done 各一个 u64） | SHMEM 三段 flag 区（send→remote_recv→recv）+ notify_ids；pto-isa 用 64 深度 flag payload 环 | demo 机制等价（flag SQE 排在数据 SQE 之后落地即证明排空），不支持并发多 post |
| quiet 轮询/超时：`ld_dev` 低字轮询 + 次数预算 `kQuietPollLimit` | SHMEM：`copy_gm_to_gm` + DCCI + volatile 回读，`GetSystemCycle()` 60 s 周期限额（A5：1000 cycles/µs）；pto-isa sdma：MTE2 搬运回读 | 机制等价（都绕开标量 L1）；`GetSystemCycle` 属 Ascend C API 层、无已验证的 A5 原生等价物，demo 用轮询预算规避 |
| 超时路径 `cce::printf` + `trap()` | SHMEM `aclshmemi_kernel_abort` = `AscendC::printf` + `trap()`（`shmemi_kernel_debug.h`） | 两者皆为编译器原生（pto-isa `debug.h` 零 include 裸用先例） |
| 供给失败直接清理 | pto-isa 在错误态卡上宁可泄漏部分资源也不析构（析构可能挂死） | demo 假设正常卡；现场排障若见挂死，参考 pto-isa 策略 |
| 单 device 单进程 | SHMEM 多 PE（对称堆/barrier） | 预取不需要对称性，任意 `aclrtMalloc` 的 GM 地址皆可 |
| `MakeU64Tensor` 成功后泄漏 backing buffer | pto-isa 用 TensorGuard 严格释放 | demo 短生命周期，进程退出回收 |
| 原生内置函数签名 | 设备侧用编译器原生层（`set_flag`/`wait_flag`/`copy_ubuf_to_gm_align_v2`/`dcci`/`ld_dev`），签名与常量随编译器版本和架构有差异（如 a2a3 为 7 参 `copy_gm_to_ubuf`，A5 为 `_align_v2` 变体） | 本文形式逐字取自 pto-isa `hns_1825_backend.hpp`（A5）；若编译报签名不匹配，以现场编译器内置层为准 |
| 64MB 上限未处理 | pto-isa `kSingleSqeBlockBytes=64MB`，超过按 `queue_num` 拆多条 SQE（`SubmitCmoPrefetchSqes`） | demo 每次预取 ≤64MB（8MB）无需拆分 |

**逐字段对照**：demo 的 `StarsV2Sqe` 与 SHMEM `stars_v2_sdma_cmo_sqe_t`、pto-isa
`BatchWriteItem`（A5 分支）字节布局一致；`submit_sqe` 的字段赋值与 pto-isa 修复分支
`AddOneCmoSqe`/`AddOneMemcpySqe`（全字段显式初始化、`qos=6`、`kernel_credit=254`、
`sssv/dssv/sns/dns=1`）一致。任何一处字段值的疑问，回溯这两个实现即可。
