cmake_minimum_required(VERSION 3.16.0)
project(mte2_bench CXX)

# Pass -DSOC_VERSION=<soc> for a non-950PR board, e.g. ascend950dt_9582.
set(SOC_VERSION "ascend950pr_9579" CACHE STRING "SoC version")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "build type")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN path")
if(NOT ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()
set(RUN_MODE "npu")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install")

set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake")
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(mte2_bench_kernel STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_mte2.cpp
)

add_executable(mte2_bench ${CMAKE_CURRENT_SOURCE_DIR}/main_mte2.cpp)
add_dependencies(mte2_bench mte2_bench_kernel)

target_include_directories(mte2_bench PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_BINARY_DIR}/include/mte2_bench_kernel
)

target_link_directories(mte2_bench PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(mte2_bench PRIVATE
    mte2_bench_kernel
    ascendcl
    runtime
    dl
)
