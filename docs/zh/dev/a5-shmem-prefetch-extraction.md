# A5 (Ascend950) SHMEM 预取代码完整实现路径剥离文档

版本：v1.0（2026-09-16）
源码基线：`https://atomgit.com/cann/shmem` master @ `73064fa`（CMO 适配 PR #459 已合入）
分析范围：A5/Ascend950（编译宏 `__NPU_ARCH__ == 3510`，arch 族 dav-3510 / dav-c310）的 L2 预取（prefetch）能力——三条路径的完整接口、调用栈、底层机制、环境准备与最小可用程序。

> ⚠️ 关于 "arc type 3150"：SHMEM 源码中 A5 的门控宏是 **`__NPU_ARCH__ == 3510`**（见 `src/device/gm2gm/engine/shmemi_device_sdma.h:139`），编译 flag 为 `--cce-aicore-arch=dav-c310`（族名 c310）。若你的环境某处报 "3150"，指的应是同一代 Ascend950（c310 族 → `__NPU_ARCH__=3510`），本文统一以 **3510/dav-c310** 表述。

---

## 1. 总览：A5 上的三条预取路径

| # | 路径 | 接口 | 依赖 | 适用场景 |
|---|---|---|---|---|
| A | **Host 侧预取**（流序） | `aclrtCmoAsync(ptr, size, ACL_RT_CMO_TYPE_PREFETCH, stream)` | 仅 CANN runtime（**不需要 SHMEM**） | host 已知访问模式，在流上提前预取 |
| B | **Device 侧单核预取**（固定 QP0） | `aclshmemx_cmo_nbi<T>()` + `aclshmemx_sdma_quiet<T>()` | SHMEM + SDMA 引擎 | 内核内 0 号 AIV 下发一次预取 |
| C | **Device 侧多核预取**（显式 QP） | `aclshmemx_cmo_qp_nbi<T>()` + `aclshmemx_sdma_qp_quiet<T>()` | 同上 | 每个 AIV 用独立 QP 并发下发（qp_idx = AIV 全局编号） |

**A5 能力边界**（重要）：
- 仅支持 **CMO 预取**（`ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH = 6`）。枚举虽有 WRITEBACK/INVALID/FLUSH，但 `aclshmemx_cmo_nbi` 只接受 PREFETCH，其他类型**直接 return 不下发**（`shmem_device_sdma.hpp:748`）。
- **不支持 SDMA put/get 数据搬运**：`__NPU_ARCH__==3510` 下 `aclshmemx_sdma_put_nbi` / `aclshmemx_sdma_qp_put_nbi` 直接 `aclshmemi_kernel_abort`（`shmem_device_sdma.hpp:501/521/572/604`）。
- A5 走 **STARS v2 SQE 布局 + doorbell 偏移 0x0**；A2A3（`__NPU_ARCH__==2201`）走 v1 布局 + 偏移 0x8。编译期由 `ACLSHMEM_STARS_V2_LAYOUT` 常量选择（`shmemi_device_sdma.h:139-145`）。

---

## 2. 环境准备（950 CMO 的硬性要求）

### 2.1 硬件/固件/软件版本矩阵（docs/quickstart.md §4.3）

| 项 | 要求 |
|---|---|
| SoC | Ascend950 |
| HDK 固件 | 25.5.6 及以上 |
| CANN toolkit | **9.1.0 及以上**（尝鲜版；9.0.0 无 950 CMO 支持） |
| ops 包 | **cann-950-ops-legacy 9.1.0**（toolkit 与 ops-legacy 装同一目录） |
| OS 架构 | aarch64 / x86_64 |

下载地址（来自仓内 PR #459 README）：
- toolkit 9.1.0：`https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-toolkit_9.1.0_linux-x86_64.run`（aarch64 同目录）
- 950 ops-legacy：`https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/package/master/20260612/x86_64/cann-950-ops-legacy_9.1.0_linux-x86_64.run`（aarch64 同目录 `/aarch64/`）

### 2.2 安装步骤

```bash
export INSTALL_PATH=/home/user/ascend        # 自定义，可改
chmod +x Ascend-cann-toolkit_9.1.0_linux-$(uname -m).run
chmod +x cann-950-ops-legacy_9.1.0_linux-$(uname -m).run
./Ascend-cann-toolkit_9.1.0_linux-$(uname -m).run --install --install-path=${INSTALL_PATH}
./cann-950-ops-legacy_9.1.0_linux-$(uname -m).run --install-path=${INSTALL_PATH}
source ${INSTALL_PATH}/ascend-toolkit/set_env.sh
```

### 2.3 环境自检（判据）

```bash
npu-smi info                                    # 950 在列、无报错
which bisheng || which ccec                     # CANN 编译器在 PATH
echo $ASCEND_HOME_PATH                          # 指向安装目录
# SDMA 供给的硬前提：AICPU 算子符号存在
nm -D $ASCEND_HOME_PATH/lib64/libopapi.so | grep -i ShmemSdmaStarsQuery
#  期望看到 aclnnShmemSdmaStarsQuery / ...GetWorkspaceSize
```

### 2.4 SHMEM 库构建

```bash
git clone https://atomgit.com/cann/shmem.git
cd shmem
bash scripts/build.sh -examples -soc_type Ascend950
# 产物：build/bin/cmo（示例）、build/lib/libshmem.so（运行时库）
```

构建系统映射（根 `CMakeLists.txt:154-199`，`SOC_TYPE=Ascend950`）：
- 后端名：`SHMEM_BACKEND_NAME=950`
- **新版编译器（bisheng，-xasc）**：`--npu-arch=dav-3510`
- **旧版编译器（ccec，-xcce）**：`--cce-aicore-arch=dav-c310`（vec 变体 `dav-c310-vec`），附带
  `-mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-record-overflow=true -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-dcci-insert-for-scalar=false`、`-Xhost-start -ftrapv -Xhost-end`
- 两种模式都定义 `__NPU_ARCH__=3510` → 走 STARS v2 代码路径

---

## 3. Host 侧接口与调用时序（新程序只需要这些）

引用头：`#include "shmem.h"` + `#include "acl/acl.h"`（示例 `examples/cmo/main.cpp:789-833`）。

```cpp
// ========== 初始化 ==========
int32_t device_id = my_pe % g_npus + first_npu;
aclInit(nullptr);
aclrtSetDevice(device_id);
aclrtStream stream;
aclrtCreateStream(&stream);

// ========== SHMEM 初始化（SDMA 引擎） ==========
uint64_t local_mem_size = 1024UL * 1024 * 1024;       // 对称堆大小
aclshmemx_init_attr_t attributes = {};                 // include/host_device/shmem_common_types.h
// （多 PE 才需要）attributes.ip_port / my_pe / n_pes / comm_args(uniqueid)
attributes.option_attr = {(1 << 16) + sizeof(aclshmemx_init_attr_t),
                          ACLSHMEM_DATA_OP_MTE, /*timeout*/ DEFAULT_TIMEOUT, ...};
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;   // ★ 启动 SDMA 引擎（必设）
aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, qp_num);  // ★ QP 数，1..72（ACLSHMEM_MAX_AIV_PER_NPU）
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);        // ★ 触发底层供给（见 §4）

// ========== 数据准备 ==========
void* buf;
aclrtMalloc(&buf, size, ACL_MEM_MALLOC_HUGE_FIRST);

// ========== 下发含预取的内核 ==========
my_kernel<<<block_dim, nullptr, stream>>>(reinterpret_cast<uint8_t*>(buf), size);
aclrtSynchronizeStream(stream);
aclshmem_barrier_all();        // 多 PE 同步（单 PE 可省）

// ========== 收尾 ==========
aclshmem_finalize();
aclrtDestroyStream(stream);
aclrtResetDevice(device_id);
aclrtFinalize();
```

**host 侧 API 清单**（声明于 `include/host/init/shmem_host_init.h` / `include/host/shmem_host_def.h`）：

| API | 位置 | 说明 |
|---|---|---|
| `aclshmemx_set_qp_num(engine, qp_num)` | shmem_host_init.h:140 | SDMA QP（STARS 流）数量，上限 72 |
| `aclshmemx_init_attr(bootstrap, attributes)` | shmem_host_init.h:152 | 主初始化；`data_op_engine_type & ACLSHMEM_DATA_OP_SDMA(0x02)` 时走 SDMA 供给 |
| `aclshmem_barrier_all()` | — | PE 间同步 |
| `aclshmem_finalize()` | — | 释放 |
| （备选）`aclrtCmoAsync(ptr, size, ACL_RT_CMO_TYPE_PREFETCH, stream)` | CANN acl_rt.h | **路径 A**：纯 host 预取，流序，不依赖 SHMEM |

---

## 4. Host 侧底层供给链路（`aclshmemx_init_attr` 内部做了什么）

调用链：
```
aclshmemx_init_attr()                                    [src/host/init/shmem_init.cpp:671 case ACLSHMEM_DATA_OP_SDMA]
  → 引擎位掩码 HYBM_DOP_TYPE_DEVICE_SDMA                 [src/host/init/backends/shmem_init_backend.cpp:247]
  → TransportManager->OpenDevice()                       [src/host/entity/mem_entity_default.cpp:905]
    → SdmaTransportManager::OpenDevice()                 [src/host/transport/device_sdma/device_sdma_transport_manager.cpp:37]
```

`OpenDevice` 五步（这是**设备侧能工作的全部前提**）：

| 步 | 函数（device_sdma_transport_manager.cpp） | 做什么 |
|---|---|---|
| 1 | `CreateStarsStreams(qp_num)` :93 | 每 QP：`aclrtCreateStreamWithConfig(&s, 0, ACL_STREAM_DEVICE_USE_ONLY=0x20)` → `aclrtStreamGetId` → `DlRtApi::RtStreamGetSqid/RtStreamGetCqid`（dlsym 自 runtime 的私有接口）→ `RtGetDeviceInfo(dev,0,19=phy_die_id)`；存 `host_stream_info_t{stream_, ctx_, stream_id, sq_id, cq_id, logic_cq_id, dev_id}` |
| 2 | `MallocSdmaWorkspace(28KB)` :154 | `aclrtMalloc(HUGE_FIRST)` + 清零 → 记入全局 `g_state.sdma_workspace_addr` |
| 3 | `CreateNotifyIds(qp_num)` :71 | 每 QP：`aclrtCreateNotify` + `aclrtGetNotifyId`，notify id 数组 H2D 到 `workspace + 14KB` 处 |
| 4 | `CopyHostOpResToDevice()` :164 | `streams[]` 与 `op_res_info{size, streams_addr, workspace_addr}` H2D |
| 5 | `LaunchSdmaAicpuKernel()` :219 | 建 AICPU 流（`ACL_STREAM_FAST_LAUNCH|FAST_SYNC`）；构造输入 aclTensor=`[streams_addr, workspace_addr]`、输出 aclTensor；**二段式调用 dlsym 自 `libopapi.so` 的 AICPU 算子**：`AclnnShmemSdmaStarsQueryGetWorkspaceSize(input, output, &ws, &executor)` → `AclnnShmemSdmaStarsQuery(ws, ws_size, executor, stream)` → `aclrtSynchronizeStream`。**该算子按流信息编程 STARS 通道，把 `stars_channel_flag_info_t` + `stars_channel_info_t[]` 写进 workspace** |

dlsym 细节（`src/host/utils/under_api/dl_opapi_api.cpp:32-41`）：
```cpp
dlopen("libopapi.so", RTLD_NOW)
dlsym("aclnnShmemSdmaStarsQueryGetWorkspaceSize")
dlsym("aclnnShmemSdmaStarsQuery")
```

**设备如何找到 workspace**：host 同时把 `aclshmem_device_host_state_t`（含 `sdma_workspace_addr` 字段，`shmem_common_types.h:382-431`）写入设备**固定 GM 魔法地址**：
```
SMEM_SHM_DEVICE_END_ADDR        = 0x180000000000 - 1GB  = 0x17C00000000
SMEM_SHM_DEVICE_META_ADDR       = 0x17BFE000000          (END - 32MB)
SMEM_SHM_DEVICE_USER_CONTEXT_ADDR = 0x17BFE010000        (META + 64KB)  ← device_state(shmemId=0)
```
（`src/device/shmemi_device_meta.h:15-26`；设备侧经 `aclshmemi_get_state()` → `aclshmemi_get_extra_context_addr(0)` 读取，`src/device/shmemi_device_common.hpp:43-49`）

---

## 5. Device 侧接口（内核内调用，只需这几个）

引用头：`#include "shmem.h"`（其内已含 `device/gm2gm/engine/shmem_device_sdma.h` 公共声明）。
公共 API 签名与完整注释见 `include/device/gm2gm/engine/shmem_device_sdma.h:206-324`。

```cpp
// ---- 路径 B：单核（固定 QP0，仅 0 号 AIV 可调） ----
template <typename T> ACLSHMEM_DEVICE void aclshmemx_cmo_nbi(
    __gm__ T* src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
    __ubuf__ T* buf, uint32_t ub_size, uint32_t sync_id);
template <typename T> ACLSHMEM_DEVICE void aclshmemx_sdma_quiet(
    __ubuf__ T* buf, uint32_t ub_size, uint32_t sync_id);

// ---- 路径 C：多核（显式 QP，任意 AIV，qp_idx 须与 quiet 配对） ----
template <typename T> ACLSHMEM_DEVICE void aclshmemx_cmo_qp_nbi(
    __gm__ T* src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
    __ubuf__ T* buf, uint32_t ub_size, uint32_t qp_idx, uint32_t sync_id);
template <typename T> ACLSHMEM_DEVICE void aclshmemx_sdma_qp_quiet(
    __ubuf__ T* buf, uint32_t ub_size, uint32_t qp_idx, uint32_t sync_id);
```

**参数与约束**（来自公共头注释 + 示例）：
- `src`：本端 GM 地址（`elem_size * sizeof(T)` 为操作字节数，须 ≤ UINT32_MAX）
- `cmo_type`：只传 `ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH`（=6），否则静默不下发
- `buf/ub_size`：UB 临时工作区，**≥64B、64B 对齐**（`UB_ALIGN_SIZE_64`）；示例用法：`__ubuf__ uint8_t* tmp_buff = (__ubuf__ uint8_t*)uint64_t(1024); ub_size=64;`（UB 偏移 1024 处）
- `sync_id`：管线硬件事件 ID，示例用 `EVENT_ID0`
- `qp_idx`：**须小于 `aclshmemx_set_qp_num` 配置的通道数**；路径 C 中示例 `qp_idx = AscendC::GetBlockIdx()`（全局 AIV 编号）
- 语义：`nbi` = 非阻塞提交；必须配对同 QP 的 `quiet` 才算完成；quiet 只排空对应 QP，不会排其他 QP
- 附送工具：`aclshmemx_sdma_qp_notify_record(buf, ub_size, qp_idx, sync_id)` —— 在 QP 上追加 notify-record SQE，host 侧可用 `aclrtWaitAndResetNotify` 等待（`shmem_device_sdma.h:326-375`）

---

## 6. Device 侧完整调用栈（含底层硬件接口）

### 6.1 提交路径（cmo_nbi）

```
aclshmemx_cmo_nbi<T>()                          [src/device/gm2gm/engine/shmem_device_sdma.hpp:745]
aclshmemx_cmo_qp_nbi<T>()                       [:779]（多核变体，多一个 qp_idx）
  ├─ if (cmo_type != CMO_TYPE_PREFETCH) return;     // 只支持预取
  ├─ LocalTensor ub_tensor{VECOUT, buf, ub_size}
  └─ aclshmemi_cmo_async(src, size, cmo_type, ub_tensor, sync_id, qp_idx)   [:456]
       ├─ aclshmemi_get_state()                     [src/device/shmemi_device_common.hpp:43]
       │    └─ 读固定 GM 地址 0x17BFE010000 的 device_state
       │       → device_state->sdma_workspace_addr（host 供给的 28KB workspace）
       ├─ channel_base = workspace + 64B（跳过 flag_info 头）
       ├─ channel_info = channel_base + qp_idx × 64B（stars_channel_info_t）
       ├─ dcci_cacheline(channel_info + 4)          [src/device/gm2gm/shmemi_device_cc.h:511]
       │    └─ AscendC::DataCacheCleanAndInvalid<SINGLE_CACHE_LINE, CACHELINE_OUT>
       ├─ sq_tail = *(u32*)(channel_info + 4)       // channel_info->sq_tail
       ├─ aclshmemi_cmo_submit_data_sqes(...)       [:442]
       │    └─ if constexpr (ACLSHMEM_STARS_V2_LAYOUT)   // __NPU_ARCH__==3510
       │         aclshmemi_fill_stars_v2_cmo_sqe(channel_info, src, size, opcode=6, sq_tail, task_id)  [:193]
       │           ├─ sqe = (stars_v2_sdma_cmo_sqe_t*)channel_info->sq_base
       │           │        + (sq_tail % channel_info->sq_depth)     // 64B SQE 槽
       │           ├─ header: type=11(ACLSHMEM_SQE_TYPE_SDMA), l1_lock/l1_unlock/ie/pre_p/post_p=0,
       │           │         wr_cqe=1, ptr_mode/rtt_mode/head_update=0, block_dim=0,
       │           │         rt_streamid=stream_id, task_id=(uint16)(sq_tail-sq_head)
       │           ├─ res3/res4=0, kernel_credit=254, ptr_mode/res5=0
       │           ├─ opcode=cmo_type(6=PREFETCH), sssv/dssv/sns/dns=1, sro/dro/stride/ie2/comp_en/res6=0
       │           ├─ sqe_id=0, mpam_partid/mpamns/pmg=0, qos=6 (HCCL QoS), res7=0
       │           ├─ src/dst_streamid、sub_streamid 全 0
       │           ├─ src_addr_low/high = src 拆 32 位；dst_addr=0
       │           └─ length=size, src/dst_stride_len=0, stride_num=0
       │       （v1 分支 aclshmemi_fill_cmo_sqe [:258]：A2A3 用，wr_cqe=0、qos=6、partid=63）
       ├─ sq_tail = (sq_tail + 1) % channel_info->sq_depth
       ├─ dcci_cachelines(sqe_slot, 64B)            [cc.h:523]（按 cache line 逐行清出）
       ├─ aclshmemi_set_value<u32>(sq_reg_base + 0x0, sq_tail)   ★ Ring Doorbell（V2 偏移 0x0）
       └─ aclshmemi_set_value<u32>(channel_info + 4, sq_tail)    ★ 镜像 tail 供下次读
            └─ [shmem_device_sdma.hpp:61] tmp_local.SetValue(0,x)
               → SetFlag<HardEvent::S_MTE3> → DataCopyPad(UB→GM 4B) → MTE3_S 等待
```

### 6.2 完成路径（quiet）

```
aclshmemx_sdma_quiet<T>(buf, ub_size, sync_id)        [:662]（QP0）
aclshmemx_sdma_qp_quiet<T>(buf, ub_size, qp_idx, sync_id)  [:683]（显式 QP）
  └─ aclshmemi_sdma_quiet(ub_tensor, qp_idx, sync_id)      [:642]
       ├─ channel_info = channel_base + qp_idx
       ├─ flag 区布局：
       │    flag_base = workspace + 14KB(NOTIFY_ADDR_OFFSET) + 72×4(notify_ids)
       │    layout.send_workspace        = flag_base + 64B × qp_idx
       │    layout.remote_recv_workspace = send + channel_num × 64B
       │    layout.recv_workspace        = remote_recv + channel_num × 64B
       ├─ aclshmemi_sdma_submit_flag_sqes(...)              [:335]
       │    ├─ aclshmemi_set_value(send_workspace, 1)       // 先布防 flag
       │    ├─ 读 sq_tail（先 dcci）
       │    ├─ V2: aclshmemi_fill_stars_v2_sdma_sqe(send → remote_recv, 8B)   [:114]
       │    │      （复用 64B v2 SQE，opcode=0，一次 8 字节 SDMA 写回 flag）
       │    ├─ dcci_cachelines(sqe_slot, 64B)
       │    └─ 双写 doorbell（sq_reg_base+0x0 与 channel_info+4）
       └─ aclshmemi_sdma_poll_for_completion(layout, ...)   [:373]
            ├─ 轮询：copy_gm_to_gm(recv ← remote_recv, 4B) + dcci_cacheline(recv)
            │        直到 recv 非 0（950 计时 1000 cycles/us，超时 60s）
            ├─ 超时 → 复位两 flag → aclshmemi_kernel_abort（printf+trap）
            └─ 正常 → 两 flag 复位 0
```

> 注意：quiet 的 flag 机制用的是 8B **SDMA 写** SQE——这是 950 上被支持的内部机制（对外 put/get 才被禁）。

### 6.3 底层硬件原语清单（内核内直接可用）

| 原语 | 来源 | 用途 |
|---|---|---|
| `AscendC::DataCacheCleanAndInvalid<T, CacheLine, DcciDst::CACHELINE_OUT>` | kernel_operator.h | DCCI：SQE/元数据写后清 cache |
| `AscendC::DataCopyPad` + `SetFlag/WaitFlag<HardEvent::S_MTE3 / MTE3_S / MTE2_MTE3>` | 同上 | UB↔GM 搬运与管线同步（set_value/copy_gm_to_gm 的基础） |
| `AscendC::GetSystemCycle()` | 同上 | 轮询计时（950: 1000 cycles/us） |
| `AscendC::GetBlockIdx()/GetBlockNum()`、`ASCEND_IS_NOT_AIV` | 同上 | AIV 门控（CMO 只在 AIV 上执行，AIC 直接 return） |
| `trap()` + `AscendC::printf` | kernel debug | 越界/超时 abort |

---

## 7. 关键数据结构与常量（可直接搬走）

### 7.1 通道与 SQE 结构（`shmemi_device_sdma.h`）

```cpp
struct stars_channel_flag_info_t {            // 64B，workspace 头
    uint32_t flag; uint32_t totalQueueNum; uint8_t reserved[56];
};
struct stars_channel_info_t {                 // 64B/通道，workspace[64B..]
    uint32_t sq_head;      // +0
    uint32_t sq_tail;      // +4   ← 设备侧读写 tail 的镜像
    uint64_t sq_base;      // +8   SQ 缓冲区基地址
    uint64_t sq_reg_base;  // +16  doorbell 寄存器基地址
    uint32_t sq_depth;     // +24
    uint32_t sq_id;        // +28
    uint32_t cq_id;        // +32
    uint32_t logic_cq_id;  // +36
    uint64_t cqe_addr;     // +40
    uint32_t report_cqe_num; // +48
    uint32_t stream_id;    // +52
    uint32_t dev_id;       // +56
    uint8_t  reserved[4];  // +60
};

enum class ACLSHMEMCMOTYPE : uint32_t {
    CMO_TYPE_PREFETCH = 6,   // GM → L2（唯一被接受的值）
    CMO_TYPE_WRITEBACK, CMO_TYPE_INVALID, CMO_TYPE_FLUSH, CMO_TYPE_MAX,
};

// A5（__NPU_ARCH__==3510）：V2 布局 + doorbell 偏移 0
constexpr bool     ACLSHMEM_STARS_V2_LAYOUT = true;       // 3510 时
constexpr uint32_t ACLSHMEM_STARS_SQ_TAIL_OFFSET = 0x0;   // 3510 时（v1/A2A3 为 0x8）

struct stars_v2_sdma_cmo_sqe_t {              // 64B（v2，A5）
    stars_v2_sqe_header_t header;             // type:6|l1_lock|l1_unlock|ie|pre_p|post_p|wr_cqe|ptr_mode
                                              // |rtt_mode|head_update|reserved, block_dim:16,
                                              // rt_streamid:16, task_id:16
    uint32_t res3; uint16_t res4;             // 8~15
    uint8_t kernel_credit;                    // =254
    uint8_t ptr_mode:1, res5:7;               // 15
    uint32_t opcode:8, sssv:1, dssv:1, sns:1, dns:1, sro:1, dro:1, stride:2,  // 16~19
             ie2:1, comp_en:1, res6:14;
    uint16_t sqe_id; uint8_t mpam_partid;     // 20~23
    uint8_t mpamns:1, pmg:2, qos:4, res7:1;   // qos=6 (HCCL)
    uint16_t src_streamid, src_sub_streamid;  // 24~31
    uint16_t dst_streamid, dst_sub_streamid;
    uint32_t src_addr_low, src_addr_high;     // 32~47
    uint32_t dst_addr_low, dst_addr_high;
    uint32_t length, src_stride_len, dst_stride_len, stride_num;  // 48~63
};
```

### 7.2 常量表

| 常量 | 值 | 出处 |
|---|---|---|
| `ACLSHMEM_SQE_TYPE_SDMA` | 11 | shmem_device_sdma.hpp:25 |
| `ACLSHMEM_SQE_TYPE_NOTIFY_RECORD` | 6 | :26 |
| `ACLSHMEM_STARS_DEFAULT_KERNEL_CREDIT` | 240 | :27（v1） |
| `ACLSHMEM_DEFAULT_KERNEL_CREDIT` | 254 | :28（**v2/A5 用**） |
| `ACLSHMEM_STARS_NOTIFY_ADDR_OFFSET` | 14KB | shmem_common_types.h:169 |
| `ACLSHMEM_SDMA_FLAG_LENGTH` | 64 | :173 |
| `ACLSHMEM_SDMA_WORKSPACE_SIZE` | 28KB | :178 |
| `ACLSHMEM_MAX_AIV_PER_NPU` | 72 | :236 |
| `UB_ALIGN_SIZE_64` | 64 | device/shmem_def.h:24 |

### 7.3 28KB workspace 内存布局

```
+0x0000   stars_channel_flag_info_t (64B: flag, totalQueueNum)
+0x0040   stars_channel_info_t[72]（每通道 64B）
+0x3800   notify_ids[72]（u32，14KB 处）
+0x3920   flag 区（3 段 × channel_num × 64B）：
            send[qp] → remote_recv[qp] → recv[qp]
          （send 布防 1 → flag SQE 8B 写到 remote_recv → 轮询 copy 回 recv 读）
```

---

## 8. 最小可用程序模板（照抄即可跑）

### 8.1 程序（`my_cmo.cpp`）

```cpp
#include <cstdio>
#include "shmem.h"
#include "acl/acl.h"
#include "kernel_operator.h"

#define CHECK_RET(f) do { int r = (f); if (r != 0) { printf("%s:%d err %d\n", __FILE__, __LINE__, r); return r; } } while (0)

// ---------- device kernel（路径 B：QP0 单核） ----------
__global__ __aicore__ void prefetch_once(GM_ADDR src, uint32_t size)
{
    if ASCEND_IS_NOT_AIV { return; }                 // 仅 AIV
    if (AscendC::GetBlockIdx() != 0) { return; }     // QP0 接口只允许 0 号 AIV
    constexpr uint32_t ub_offset = 1024;             // UB 内 64B 临时区（64B 对齐）
    constexpr uint32_t ub_size = 64;
    __ubuf__ uint8_t* tmp_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(ub_offset));
    aclshmemx_cmo_nbi(reinterpret_cast<__gm__ uint8_t*>(src), size,
                      ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, EVENT_ID0);
    aclshmemx_sdma_quiet(tmp_buff, ub_size, EVENT_ID0);
}

// ---------- device kernel（路径 C：每 AIV 一个 QP） ----------
__global__ __aicore__ void prefetch_per_aiv(GM_ADDR src, uint32_t size, uint32_t aiv_num)
{
    if ASCEND_IS_NOT_AIV { return; }
    const uint32_t block_id = AscendC::GetBlockIdx();
    if (block_id >= aiv_num) { return; }
    constexpr uint32_t ub_offset = 1024, ub_size = 64;
    __ubuf__ uint8_t* tmp_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(ub_offset));
    __gm__ uint8_t* my_region = reinterpret_cast<__gm__ uint8_t*>(src) + (uint64_t)size * block_id;
    aclshmemx_cmo_qp_nbi(my_region, size, ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH,
                         tmp_buff, ub_size, /*qp_idx=*/block_id, EVENT_ID0);
    aclshmemx_sdma_qp_quiet(tmp_buff, ub_size, /*qp_idx=*/block_id, EVENT_ID0);
}

// ---------- host ----------
int main(int argc, char* argv[])
{
    const int n_pes = atoi(argv[1]);        // PE 总数
    const int my_pe = atoi(argv[2]);        // 本 PE
    const char* ipport = argv[3];           // 如 tcp://127.0.0.1:8766
    const int g_npus = atoi(argv[4]);       // NPU 数
    const int f_pe = atoi(argv[5]);         // 首 PE 号
    const int f_npu = atoi(argv[6]);        // 首 NPU 号

    const uint32_t qp_num = 4;              // ≥1 且 ≤72；路径 C 需要 ≥ aiv_num
    const uint32_t prefetch_mb = 8;         // 预取 8MB
    const uint32_t size = prefetch_mb * 1024 * 1024;

    int32_t device_id = my_pe % g_npus + f_npu;
    CHECK_RET(aclInit(nullptr));
    CHECK_RET(aclrtSetDevice(device_id));
    aclrtStream stream;
    CHECK_RET(aclrtCreateStream(&stream));

    // ---- SHMEM init（SDMA 引擎） ----
    aclshmemx_init_attr_t attributes = {};
    attributes.my_pe = my_pe;
    attributes.n_pes = n_pes;
    snprintf(attributes.ip_port, sizeof(attributes.ip_port), "%s", ipport);
    attributes.local_mem_size = 1024UL * 1024 * 1024;
    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes.option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, -1, -1, -1};  // timeout 用默认值
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;         // ★
    CHECK_RET(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, qp_num));             // ★
    CHECK_RET(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));   // ★ 触发 §4 供给链

    // ---- 数据 ----
    void* src_ptr = nullptr;
    CHECK_RET(aclrtMalloc(&src_ptr, (size_t)size * qp_num, ACL_MEM_MALLOC_HUGE_FIRST));

    // ---- 路径 A（可选对照）：纯 host 预取 ----
    // CHECK_RET(aclrtCmoAsync(src_ptr, size, ACL_RT_CMO_TYPE_PREFETCH, stream));

    // ---- 路径 B：单核 QP0 预取 ----
    prefetch_once<<<1, nullptr, stream>>>(reinterpret_cast<GM_ADDR>(src_ptr), size);
    CHECK_RET(aclrtSynchronizeStream(stream));

    // ---- 路径 C：qp_num 个 AIV 各自 QP 预取 ----
    prefetch_per_aiv<<<(qp_num + 1) / 2, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(src_ptr), size, qp_num);
    CHECK_RET(aclrtSynchronizeStream(stream));

    aclshmem_barrier_all();

    CHECK_RET(aclrtFree(src_ptr));
    CHECK_RET(aclshmem_finalize());
    CHECK_RET(aclrtDestroyStream(stream));
    CHECK_RET(aclrtResetDevice(device_id));
    CHECK_RET(aclFinalize());
    printf("[SUCCESS] pe %d\n", my_pe);
    return 0;
}
```

### 8.2 编译（两种方式）

**方式一：挂进 SHMEM 示例构建（推荐，零配置）**
```bash
# examples/mycmo/CMakeLists.txt：
#   aclshmem_add_fusion_example(mycmo my_cmo.cpp)
# 并把 mycmo 加入 examples/CMakeLists.txt 的子目录列表，然后：
bash scripts/build.sh -examples -soc_type Ascend950
# 产物 build/bin/mycmo
```
该函数自动加上（`examples/CMakeLists.txt:41`）：`-xasc/-xcce + npu-arch/aicore-arch`、`--cce-fatobj-link`、头路径（`include/ src/device/ examples/utils` 等）、链接 `libshmem`。

**方式二：独立编译（ccec 命令行，与 pto-isa 验证过的同族命令）**
```bash
ccec -c -O2 -x cce -std=c++17 --cce-aicore-only \
  -mllvm -cce-aicore-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false \
  --cce-aicore-arch=dav-c310 \
  -I<shmem>/include -I<shmem>/src/device -I$ASCEND_HOME_PATH/include \
  -o my_cmo.o my_cmo.cpp
# 链接 host 可执行：g++ ... --cce-fatobj-link -L<shmem>/build/lib -lshmem
```

### 8.3 运行

```bash
export SHMEM_UID_SESSION_ID=127.0.0.1:8899
export LD_LIBRARY_PATH=<shmem>/build/lib:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH
# 单机单卡（n_pes=1, g_npus=1）：
./build/bin/mycmo 1 0 tcp://127.0.0.1:8766 1 0 0
# 多卡（8 NPU、8 PE）：
for i in $(seq 0 7); do ./build/bin/mycmo 8 $i tcp://127.0.0.1:8766 8 0 0 & done; wait
```

---

## 9. 与 pypto/pto-isa 侧实现的对应关系（交叉参考）

| SHMEM (cann/shmem master) | pto-isa (fix 分支) | 说明 |
|---|---|---|
| `aclshmemi_fill_stars_v2_cmo_sqe` (shmem_device_sdma.hpp:193) | `AddOneCmoSqe` A5 分支 (sdma_cmo_intrin.hpp:35) | 我们已将 pto-isa 补齐到与 SHMEM 字节级一致（qos=6、全字段初始化） |
| `ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH = 6` | `kCmoPrefetchOpcode = 6U` | 同一硬件 opcode |
| `stars_channel_info_t` | `BatchWriteChannelInfo` | 同一 64B 布局 |
| `ACLSHMEM_STARS_SQ_TAIL_OFFSET = 0x0` (3510) | v2 doorbell `sq_reg_base+0` | 同一 doorbell 偏移 |
| `ACLSHMEM_DEFAULT_KERNEL_CREDIT = 254` | A5 `kCreditTimeDefault=254` | 同一 credit |
| `SdmaTransportManager::OpenDevice` 五步 + `aclnnShmemSdmaStarsQuery` | simpler `comm_hccl.cpp` 的 `SdmaWorkspaceManager::Init` 镜像实现 | 同一 host 供给机制 |

---

## 10. 源文件索引（上游仓：`https://atomgit.com/cann/shmem`，master @ `73064fa`）

| 文件 | 内容 |
|---|---|
| `examples/cmo/main.cpp` | 完整示例（三条路径 + 性能统计 + CSV 输出） |
| `include/device/gm2gm/engine/shmem_device_sdma.h` | 设备侧公共 API 声明（含完整 doxygen 约束） |
| `src/device/gm2gm/engine/shmem_device_sdma.hpp` | 设备侧实现（SQE 填充/doorbell/quiet/轮询） |
| `src/device/gm2gm/engine/shmemi_device_sdma.h` | 结构体定义 + V2 布局选择 + 常量 |
| `src/device/gm2gm/shmemi_device_cc.h` | `dcci_cacheline(s)` / `aclshmemi_set_value` 等底层原语 |
| `src/device/shmemi_device_meta.h` | 设备固定魔法地址与元数据布局 |
| `src/device/shmemi_device_common.hpp` | `aclshmemi_get_state()` |
| `src/host/transport/device_sdma/device_sdma_transport_manager.cpp` | host 供给链（STARS 流 + AICPU 算子） |
| `src/host/utils/under_api/dl_opapi_api.cpp` | libopapi dlsym |
| `src/host/init/backends/shmem_init_backend.cpp` | 引擎位选择 |
| `include/host_device/shmem_common_types.h` | `aclshmemx_init_attr_t` / device_host_state / 常量 |
| `docs/quickstart.md` / `docs/compilation_build_guide.md` | CANN 版本矩阵与构建指南 |
