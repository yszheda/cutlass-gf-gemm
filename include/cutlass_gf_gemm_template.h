/**
 * @file cutlass_gf_gemm_template.h
 * @brief CUTLASS template-based GEMM backend for GF(2^8)
 *
 * Uses CUTLASS's device::Gemm template with custom GF(2^8) numeric traits
 * and MMA operator.
 */

#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include "cutlass_gf_gemm.h"
#include "cutlass_gf28_traits.h"

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cutlass_gf_gemm_init_tables(uint8_t** d_gf_exp, uint8_t** d_gf_log);
void cutlass_gf_gemm_free_tables(uint8_t* d_gf_exp, uint8_t* d_gf_log);

GFGemmError cutlass_gf_gemm_execute(
    int m, int n, int k,
    const uint8_t* A, int lda,
    const uint8_t* B, int ldb,
    uint8_t* C, int ldc,
    const uint8_t* d_gf_exp,
    const uint8_t* d_gf_log,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
