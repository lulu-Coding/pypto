// Pure MTE2 (GM -> L1) streaming throughput.
//
// Every AIC streams an exclusive contiguous slice of a 1 GiB GM buffer into
// L1 with 8-deep multibuffering and no consumer. Aggregate bytes / wall time
// is the board's GM->L1 streaming rate. This is the reference used to decide
// whether a bandwidth-bound grouped matmul is at its board's memory floor.

#if defined(__DAV_C310__) || defined(__CCE_AICORE__)

#include "kernel_operator.h"

using namespace AscendC;

namespace mb {
constexpr int32_t CHUNK = 65536;                   // 64 KiB contiguous
constexpr int32_t NSTAGE = 8;                      // 8 * 64 KiB = 512 KiB L1
constexpr uint64_t TOTAL = 1073741824ULL;          // 1 GiB
}  // namespace mb

__global__ __aicore__ void mte2_bench(GM_ADDR src, GM_ADDR dst) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    using namespace mb;

    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    if (blockNum == 0U) {
        return;
    }

    GlobalTensor<int8_t> sGm;
    sGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(src), TOTAL);

    LocalTensor<int8_t> l1(TPosition::A1, 0, TOTAL_L1_SIZE);

    uint64_t perCore = TOTAL / blockNum;
    perCore = (perCore / CHUNK) * CHUNK;
    uint64_t start = static_cast<uint64_t>(blockIdx) * perCore;
    int32_t nchunk = static_cast<int32_t>(perCore / CHUNK);

    for (int32_t i = 0; i < nchunk; ++i) {
        int32_t s = i % NSTAGE;
        if (i >= NSTAGE) {
            // Wait for the copy that used this slot NSTAGE iterations ago.
            WaitFlag<HardEvent::MTE2_MTE1>(s);
        }
        DataCopy(l1[s * CHUNK], sGm[start + static_cast<uint64_t>(i) * CHUNK], static_cast<uint32_t>(CHUNK));
        SetFlag<HardEvent::MTE2_MTE1>(s);
    }

    int32_t drain = (nchunk < NSTAGE) ? nchunk : NSTAGE;
    for (int32_t s = 0; s < drain; ++s) {
        WaitFlag<HardEvent::MTE2_MTE1>(s);
    }
}

#endif
