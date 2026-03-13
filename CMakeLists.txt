cmake_minimum_required(VERSION 3.18)
project(cutlass-gf-gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# CUDA architecture - detect from current device or use defaults
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
endif()

# CUTLASS path - use environment CUTLASS_PATH if set, otherwise check common locations
if(DEFINED ENV{CUTLASS_PATH})
    set(CUTLASS_PATH "$ENV{CUTLASS_PATH}" CACHE PATH "Path to CUTLASS")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/../cutlass/include/cutlass/cutlass.h")
    # Check parent directory for cutlass (remote server setup)
    set(CUTLASS_PATH "${CMAKE_SOURCE_DIR}/../cutlass" CACHE PATH "Path to CUTLASS")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include/cutlass/cutlass.h")
    set(CUTLASS_PATH "${CMAKE_SOURCE_DIR}/3rdparty/cutlass" CACHE PATH "Path to CUTLASS")
else()
    set(CUTLASS_PATH "${CMAKE_SOURCE_DIR}/3rdparty/cutlass" CACHE PATH "Path to CUTLASS")
endif()

# Find CUTLASS
if(NOT EXISTS "${CUTLASS_PATH}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_PATH}. Please set CUTLASS_PATH environment variable.")
endif()

message(STATUS "Using CUTLASS from: ${CUTLASS_PATH}")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUTLASS_PATH}/include
    ${CUTLASS_PATH}/tools/util/include
)

# CUDA path
find_package(CUDAToolkit REQUIRED)

# Compiler options
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wno-deprecated-gpu-targets>
)

# Release build options
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Library target
add_library(cutlass_gf_gemm STATIC
    src/cutlass_gf_gemm.cu
)

set_target_properties(cutlass_gf_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cutlass_gf_gemm PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUTLASS_PATH}/include
)

# Example executable
add_executable(example_gf_gemm examples/example_gf_gemm.cu)
target_link_libraries(example_gf_gemm PRIVATE cutlass_gf_gemm CUDA::cudart)

# Test executable
add_executable(test_gf_gemm tests/test_gf_gemm.cu)
target_link_libraries(test_gf_gemm PRIVATE cutlass_gf_gemm CUDA::cudart)

# Install rules
install(TARGETS cutlass_gf_gemm
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES
    include/gf_ops.h
    include/cutlass_gf_gemm.h
    include/gf16.h
    DESTINATION include
)
