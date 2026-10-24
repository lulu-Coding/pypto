// Host driver for the pure GM->L1 MTE2 streaming benchmark.
//
// Usage: mte2_bench [blockDim]
//   blockDim defaults to 28. Pass the board's AIC count (e.g. 32 on
//   Ascend950DT_9582) to measure the aggregate streaming rate; a value larger
//   than the physical AIC count runs blocks in multiple waves and reports an
//   invalid aggregate.
//
// The device id comes from TASK_DEVICE (set by task-submit), default 0.

#include <acl/acl.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "aclrtlaunch_mte2_bench.h"

#define CHECK_ACL(expr)                                                                  \
    do {                                                                                 \
        aclError __ret = (expr);                                                         \
        if (__ret != ACL_SUCCESS) {                                                      \
            std::printf("[ERROR] %s failed, ret=%d at %s:%d\n", #expr, (int)__ret,       \
                        __FILE__, __LINE__);                                             \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

static constexpr uint64_t TOTAL = 1073741824ULL;  // 1 GiB

int main(int argc, char** argv) {
    uint32_t blockDim = (argc > 1) ? static_cast<uint32_t>(std::atoi(argv[1])) : 28;
    const char* td = std::getenv("TASK_DEVICE");
    int32_t deviceId = td ? static_cast<int32_t>(std::atoi(td)) : 0;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* dev = nullptr;
    CHECK_ACL(aclrtMalloc(&dev, TOTAL, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(dev, TOTAL, 1, TOTAL));

    aclrtlaunch_mte2_bench(blockDim, stream, dev, dev);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    const int RUNS = 20;
    double best = 1e18;
    for (int r = 0; r < RUNS; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        aclrtlaunch_mte2_bench(blockDim, stream, dev, dev);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (us < best) best = us;
    }

    const int BATCH = 20;
    auto b0 = std::chrono::steady_clock::now();
    for (int r = 0; r < BATCH; ++r) {
        aclrtlaunch_mte2_bench(blockDim, stream, dev, dev);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto b1 = std::chrono::steady_clock::now();
    double batchUs = std::chrono::duration<double, std::micro>(b1 - b0).count() / BATCH;

    double gbps = static_cast<double>(TOTAL) / (best * 1e-6) / 1e9;
    std::printf("MTE2 CFG device=%d blockDim=%u total=%llu bytes\n", deviceId, blockDim,
                (unsigned long long)TOTAL);
    std::printf("MTE2 TIME best=%.3f us  batch=%.3f us\n", best, batchUs);
    std::printf("MTE2 BW best=%.1f GB/s (%.3f TB/s)   batch=%.1f GB/s (%.3f TB/s)\n", gbps, gbps / 1000.0,
                static_cast<double>(TOTAL) / (batchUs * 1e-6) / 1e9,
                static_cast<double>(TOTAL) / (batchUs * 1e-6) / 1e12);

    aclrtFree(dev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
