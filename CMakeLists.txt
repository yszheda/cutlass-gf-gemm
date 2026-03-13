cmake_minimum_required(VERSION 3.18)
project(cutlass-gf-gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)

# CUTLASS path
set(CUTLASS_PATH "${CMAKE_SOURCE_DIR}/3rdparty/cutlass" CACHE PATH "Path to CUTLASS")

# Find CUTLASS
if(NOT EXISTS "${CUTLASS_PATH}")
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_PATH}. Please clone with --recursive or set CUTLASS_PATH.")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUTLASS_PATH}/include
    ${CUTLASS_PATH}/tools/util/include
)

# Compiler options
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
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
)

# Example executable
add_executable(example_gf_gemm examples/example_gf_gemm.cu)
target_link_libraries(example_gf_gemm PRIVATE cutlass_gf_gemm)

# Test executable
add_executable(test_gf_gemm tests/test_gf_gemm.cu)
target_link_libraries(test_gf_gemm PRIVATE cutlass_gf_gemm)

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
