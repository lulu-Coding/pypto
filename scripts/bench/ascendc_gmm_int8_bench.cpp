// AscendC INT8 x INT8 -> INT32 grouped-matmul reference benchmark.
//
// Calls the CANN 9.3 aclnnGroupedMatmulWeightNz operator with a per-group NZ
// weight, matching the PyPTO 10k kernel (x [M,K] INT8, weight [G,K,N] INT8
// packed as [G,N/32,K/16,16,32], out [M,N] INT32, groupType=0).
//
// Shapes are read from the environment: GMM_M, GMM_K, GMM_N, GMM_G, GMM_REPS.
// The device id comes from TASK_DEVICE (set by task-submit), default 0.
//
// Build (CANN 9.3, no ops-transformer build needed):
//   g++ -std=c++17 -O2 -I$ASCEND_HOME_PATH/include \
//       gmm_int8_bench.cpp -L$ASCEND_HOME_PATH/lib64 -Wl,-rpath,$ASCEND_HOME_PATH/lib64 \
//       -lopapi -lascendcl -lnnopbase -lpthread -ldl -o gmm_int8_bench
//
// Run (through the queue):
//   task-submit --device auto --run run_gmm_int8_ascendc.sh
//
// Measured on Ascend950PR 9579, 28 AIC @1.65GHz, 8192x3584x3072, G=32:
//   msprof kernel Task Duration: 286.6 / 288.4 / 290.9us (Block Dim 28, cube)
//   ACL event around one warmed launch: 322.0-323.4us (includes stream/launch)
//
// The process deliberately exits without ACL teardown: a consumed executor
// cannot be reused (the CANN op double-frees), and this is a one-shot
// benchmark. All resources are reclaimed when the process exits.

#include <acl/acl.h>
#include <aclnnop/aclnn_grouped_matmul_weight_nz.h>
#include <aclnnop/aclnn_npu_format_cast.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CHECK_RET(cond, expr)                          \
    do {                                               \
        if (!(cond)) {                                 \
            std::fprintf(stderr, "CHECK failed: %s\n", #cond); \
            expr;                                      \
        }                                              \
    } while (0)

static int64_t Numel(const std::vector<int64_t>& s) {
    return std::accumulate(s.begin(), s.end(), int64_t{1}, std::multiplies<int64_t>());
}

static std::vector<int64_t> RowMajorStrides(const std::vector<int64_t>& s) {
    std::vector<int64_t> st(s.size(), 1);
    for (int64_t i = static_cast<int64_t>(s.size()) - 2; i >= 0; --i) st[i] = s[i + 1] * st[i + 1];
    return st;
}

static aclTensor* MakeNdTensor(void* dev, const std::vector<int64_t>& shape, aclDataType dtype) {
    auto strides = RowMajorStrides(shape);
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), dev);
}

// Convert a logical [G, K, N] INT8 ND tensor to the NZ physical layout.
// The returned tensor owns dstDev; the source tensor/device buffer are released.
static int CastWeightToNz(const int8_t* host, int64_t hostElems, const std::vector<int64_t>& logical,
                          aclrtStream stream, aclTensor** out, void** dstDev) {
    void* srcDev = nullptr;
    int64_t bytes = hostElems;
    aclError aclRet = aclrtMalloc(&srcDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        std::fprintf(stderr, "NZ: src aclrtMalloc(%lld) failed ret=%d msg=%s\n", (long long)bytes, (int)aclRet,
                     aclGetRecentErrMsg());
        return aclRet;
    }
    aclRet = aclrtMemcpy(srcDev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        std::fprintf(stderr, "NZ: src memcpy failed ret=%d msg=%s\n", (int)aclRet, aclGetRecentErrMsg());
        return aclRet;
    }

    aclTensor* src = MakeNdTensor(srcDev, logical, ACL_INT8);
    int64_t* dstShape = nullptr;
    uint64_t dstShapeSize = 0;
    int actualFormat = 0;
    aclnnStatus ret = aclnnNpuFormatCastCalculateSizeAndFormat(
        src, 29, ACL_FORMAT_ND, &dstShape, &dstShapeSize, &actualFormat);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "FormatCastCalculateSizeAndFormat failed: %d msg=%s\n", ret, aclGetRecentErrMsg());
        return ret;
    }
    int64_t dstElems = 1;
    for (uint64_t i = 0; i < dstShapeSize; ++i) dstElems *= dstShape[i];
    aclRet = aclrtMalloc(dstDev, dstElems, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        std::fprintf(stderr, "NZ: dst aclrtMalloc(%lld) failed ret=%d msg=%s\n", (long long)dstElems, (int)aclRet,
                     aclGetRecentErrMsg());
        return aclRet;
    }

    auto logicalStrides = RowMajorStrides(logical);
    aclTensor* dst = aclCreateTensor(logical.data(), logical.size(), ACL_INT8, logicalStrides.data(), 0,
                                     static_cast<aclFormat>(actualFormat), dstShape, dstShapeSize, *dstDev);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnNpuFormatCastGetWorkspaceSize(src, dst, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "FormatCastGetWorkspaceSize failed: %d msg=%s\n", ret, aclGetRecentErrMsg());
        return ret;
    }
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclRet = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    }
    ret = aclnnNpuFormatCast(workspace, workspaceSize, executor, stream);
    if (workspace) aclrtFree(workspace);
    aclDestroyTensor(src);
    aclrtFree(srcDev);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "NpuFormatCast failed: %d msg=%s\n", ret, aclGetRecentErrMsg());
        return ret;
    }
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        std::fprintf(stderr, "NZ sync failed ret=%d msg=%s\n", (int)aclRet, aclGetRecentErrMsg());
        return aclRet;
    }
    *out = dst;
    return 0;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    auto envInt = [](const char* name, int64_t fallback) {
        const char* v = std::getenv(name);
        return v ? std::strtoll(v, nullptr, 10) : fallback;
    };
    const int64_t G = envInt("GMM_G", 2);
    const int64_t M = envInt("GMM_M", 512);
    const int64_t K = envInt("GMM_K", 256);
    const int64_t N = envInt("GMM_N", 128);
    const int64_t reps = envInt("GMM_REPS", 5);
    if (G <= 0 || M % G != 0 || K <= 0 || N <= 0) {
        std::fprintf(stderr, "bad shape G=%lld M=%lld K=%lld N=%lld\n", (long long)G, (long long)M, (long long)K,
                     (long long)N);
        return 1;
    }
    const int64_t Mg = M / G;

    const char* td = std::getenv("TASK_DEVICE");
    int32_t deviceId = td ? static_cast<int32_t>(std::atoi(td)) : 0;

    aclError aclRet = aclInit(nullptr);
    CHECK_RET(aclRet == ACL_SUCCESS, return 1);
    aclRet = aclrtSetDevice(deviceId);
    CHECK_RET(aclRet == ACL_SUCCESS, return 1);
    aclrtStream stream = nullptr;
    aclRet = aclrtCreateStream(&stream);
    CHECK_RET(aclRet == ACL_SUCCESS, return 1);

    std::printf("device=%d G=%lld M=%lld K=%lld N=%lld reps=%lld\n", deviceId, (long long)G, (long long)M,
                (long long)K, (long long)N, (long long)reps);

    // x: all ones. weight: group g filled with g+1, so out[g] = K * (g+1) * 1.
    std::vector<int8_t> xHost(static_cast<size_t>(M * K), 1);
    std::vector<int8_t> wHost(static_cast<size_t>(G * K * N), 1);
    for (int64_t g = 0; g < G; ++g) {
        std::memset(wHost.data() + g * K * N, static_cast<int>(g + 1), static_cast<size_t>(K * N));
    }
    std::vector<int32_t> outHost(static_cast<size_t>(M * N), 0);

    void* xDev = nullptr;
    aclRet = aclrtMalloc(&xDev, xHost.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return 1);
    aclRet = aclrtMemcpy(xDev, xHost.size(), xHost.data(), xHost.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return 1);
    aclTensor* x = MakeNdTensor(xDev, {M, K}, ACL_INT8);
    aclTensor* xListArr[1] = {x};
    aclTensorList* xList = aclCreateTensorList(xListArr, 1);

    void* outDev = nullptr;
    aclRet = aclrtMalloc(&outDev, outHost.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return 1);
    aclTensor* y = MakeNdTensor(outDev, {M, N}, ACL_INT32);
    aclTensor* yListArr[1] = {y};
    aclTensorList* yList = aclCreateTensorList(yListArr, 1);

    // Convert each [K, N] group separately (a single 352 MB TransData faults),
    // then concatenate the per-group NZ blocks into one stacked [G, K, N]
    // tensor. Its storage is [G, N/32, K/16, 16, 32] for INT8.
    if (K % 16 != 0 || N % 32 != 0) {
        std::fprintf(stderr, "NZ shape requires K%%16==0 and N%%32==0, got K=%lld N=%lld\n", (long long)K,
                     (long long)N);
        return 1;
    }
    int64_t groupBytes = K * N;
    void* stackedDev = nullptr;
    aclRet = aclrtMalloc(&stackedDev, groupBytes * G, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return 1);
    for (int64_t g = 0; g < G; ++g) {
        aclTensor* tmpTensor = nullptr;
        void* tmpDev = nullptr;
        int castRet = CastWeightToNz(wHost.data() + g * K * N, groupBytes, {K, N}, stream, &tmpTensor, &tmpDev);
        if (castRet != 0) {
            std::fprintf(stderr, "NZ group %lld failed\n", (long long)g);
            return 1;
        }
        aclRet = aclrtMemcpy(static_cast<char*>(stackedDev) + g * groupBytes, groupBytes, tmpDev, groupBytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            std::fprintf(stderr, "stacked copy group %lld failed ret=%d msg=%s\n", (long long)g, (int)aclRet,
                         aclGetRecentErrMsg());
            return 1;
        }
        aclrtFree(tmpDev);  // tmpTensor is not used after the copy
    }
    std::vector<int64_t> stackedLogical = {G, K, N};
    std::vector<int64_t> stackedStorage = {G, N / 32, K / 16, 16, 32};
    auto stackedStrides = RowMajorStrides(stackedLogical);
    aclTensor* weight = aclCreateTensor(stackedLogical.data(), stackedLogical.size(), ACL_INT8, stackedStrides.data(),
                                        0, static_cast<aclFormat>(29), stackedStorage.data(), stackedStorage.size(),
                                        stackedDev);
    aclTensor* wListArr[1] = {weight};
    aclTensorList* wList = aclCreateTensorList(wListArr, 1);

    std::vector<int64_t> groupHost(static_cast<size_t>(G));
    for (int64_t g = 0; g < G; ++g) groupHost[g] = (g + 1) * Mg;
    void* groupDev = nullptr;
    aclRet = aclrtMalloc(&groupDev, groupHost.size() * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return 1);
    aclRet = aclrtMemcpy(groupDev, groupHost.size() * sizeof(int64_t), groupHost.data(),
                         groupHost.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return 1);
    aclTensor* groupList = MakeNdTensor(groupDev, {G}, ACL_INT64);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
        xList, wList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, groupList, nullptr, nullptr, nullptr,
        /*splitItem=*/3, /*groupType=*/0, /*groupListType=*/0, /*actType=*/0, nullptr, /*quantGroupSize=*/0, yList,
        nullptr, nullptr, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "GroupedMatmulWeightNzGetWorkspaceSize failed: %d\n", ret);
        return 1;
    }
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclRet = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, return 1);
    }

    aclrtEvent evStart = nullptr;
    aclrtEvent evEnd = nullptr;
    aclrtCreateEvent(&evStart);
    aclrtCreateEvent(&evEnd);

    aclrtRecordEvent(evStart, stream);
    ret = aclnnGroupedMatmulWeightNz(workspace, workspaceSize, executor, stream);
    aclrtRecordEvent(evEnd, stream);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "GroupedMatmulWeightNz failed: %d\n", ret);
        return 1;
    }
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, return 1);
    float firstMs = 0.0f;
    aclrtEventElapsedTime(&firstMs, evStart, evEnd);
    std::printf("gmm_device_ms_first=%.4f\n", firstMs);

    // Every measured call gets a fresh executor: reusing a consumed one
    // double-frees inside the CANN op. The first call above is the warm-up.
    for (int64_t rep = 0; rep < reps; ++rep) {
        uint64_t ws = 0;
        aclOpExecutor* ex = nullptr;
        ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
            xList, wList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, groupList, nullptr, nullptr, nullptr,
            3, 0, 0, 0, nullptr, 0, yList, nullptr, nullptr, &ws, &ex);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "rep %lld GetWorkspaceSize failed: %d\n", (long long)rep, ret);
            return 1;
        }
        void* wsa = nullptr;
        if (ws > 0) {
            aclRet = aclrtMalloc(&wsa, ws, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(aclRet == ACL_SUCCESS, return 1);
        }
        aclrtRecordEvent(evStart, stream);
        ret = aclnnGroupedMatmulWeightNz(wsa, ws, ex, stream);
        aclrtRecordEvent(evEnd, stream);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "rep %lld GroupedMatmulWeightNz failed: %d\n", (long long)rep, ret);
            return 1;
        }
        aclRet = aclrtSynchronizeStream(stream);
        CHECK_RET(aclRet == ACL_SUCCESS, return 1);
        float ms = 0.0f;
        aclrtEventElapsedTime(&ms, evStart, evEnd);
        std::printf("gmm_device_ms_rep=%lld %.4f\n", (long long)rep, ms);
        std::fflush(stdout);
    }

    aclRet = aclrtMemcpy(outHost.data(), outHost.size() * sizeof(int32_t), outDev,
                         outHost.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return 1);
    std::printf("out[0]=%d expected=%lld out[%lld]=%d expected=%lld\n", outHost[0], (long long)(K * 1),
                (long long)(Mg * N), outHost[Mg * N], (long long)(K * 2));
    std::fflush(stdout);
    // One-shot process: skip ACL teardown (the consumed executors are not
    // reusable, and the OS reclaims the device buffers on exit).
    std::_Exit(0);
}
