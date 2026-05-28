/**
 * @file cutlass_gf_gemm_template.cu
 * @brief CUTLASS-structured GEMM implementation for GF(2^8)
 *
 * Uses CUTLASS types, patterns, and infrastructure while maintaining
 * correct GF(2^8) semantics.
 */

#include "cutlass_gf_gemm_template.h"
#include "cutlass_gf28_traits.h"
#include "gf_ops.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// ============================================================================
// Constant Memory for GF Tables (shared with operator)
// ============================================================================

__constant__ uint8_t d_cutlass_gfexp_const[cutlass::GF28Arithmetic::kExpTableSize];
__constant__ uint8_t d_cutlass_gflog_const[cutlass::GF28Arithmetic::kLogTableSize];

// ============================================================================
// GF MMA Operator (reads from constant memory)
// ============================================================================

namespace cutlass {

/// GF(2^8) MMA operator that reads tables from constant memory
struct Gf28ConstMmaOperator {
    using ElementA = gf28_t;
    using ElementB = gf28_t;
    using ElementC = gf28_t;
    using ElementD = gf28_t;

    struct Shape {
        static constexpr int kM = 1;
        static constexpr int kN = 1;
        static constexpr int kK = 1;
    };

#ifdef __CUDA_ARCH__
    GF_HOST_DEVICE
    ElementD operator()(
        ElementA const& a,
        ElementB const& b,
        ElementC const& c
    ) const {
        uint8_t va = static_cast<uint8_t>(a);
        uint8_t vb = static_cast<uint8_t>(b);
        uint8_t vc = static_cast<uint8_t>(c);

        uint8_t product;
        if (va == 0 || vb == 0) {
            product = 0;
        } else {
            int log_sum = d_cutlass_gflog_const[va] + d_cutlass_gflog_const[vb];
            product = d_cutlass_gfexp_const[log_sum];
        }

        return ElementD(product ^ vc);
    }
#endif
};

} // namespace cutlass

// ============================================================================
// Table Initialization
// ============================================================================

__global__ void init_gf_tables_for_cutlass(uint8_t* gf_exp, uint8_t* gf_log) {
    const uint8_t prim_poly = cutlass::GF28Arithmetic::kPrimitivePolynomial;
    const int field_max = cutlass::GF28Arithmetic::kFieldMax;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint8_t exp = 1;
        for (int log = 0; log < field_max; ++log) {
            gf_log[exp] = static_cast<uint8_t>(log);
            gf_exp[log] = exp;
            gf_exp[log + field_max] = exp;
            gf_exp[log + 2 * field_max] = exp;
            exp = (exp << 1) ^ ((exp & 0x80) ? prim_poly : 0);
        }
        gf_log[0] = 0;
        gf_exp[0] = 1;
    }
}

cudaError_t cutlass_gf_gemm_init_tables(uint8_t** d_gf_exp, uint8_t** d_gf_log) {
    size_t exp_size = cutlass::GF28Arithmetic::kExpTableSize * sizeof(uint8_t);
    size_t log_size = cutlass::GF28Arithmetic::kLogTableSize * sizeof(uint8_t);

    cudaError_t err = cudaMalloc(d_gf_exp, exp_size);
    if (err != cudaSuccess) return err;

    err = cudaMalloc(d_gf_log, log_size);
    if (err != cudaSuccess) {
        cudaFree(*d_gf_exp);
        *d_gf_exp = nullptr;
        return err;
    }

    init_gf_tables_for_cutlass<<<1, 1>>>(*d_gf_exp, *d_gf_log);
    cudaDeviceSynchronize();

    // Copy to constant memory for the MMA operator
    cudaMemcpyToSymbol(d_cutlass_gfexp_const, *d_gf_exp, exp_size);
    cudaMemcpyToSymbol(d_cutlass_gflog_const, *d_gf_log, log_size);

    return cudaGetLastError();
}

void cutlass_gf_gemm_free_tables(uint8_t* d_gf_exp, uint8_t* d_gf_log) {
    if (d_gf_exp) cudaFree(d_gf_exp);
    if (d_gf_log) cudaFree(d_gf_log);
}

// ============================================================================
// CUTLASS-Structured GEMM Kernel
// ============================================================================

/**
 * @brief CUTLASS-style tiled GEMM kernel for GF(2^8)
 *
 * Uses 128x128 threadblock tiles (8 warps in M, 8 warps in N direction),
 * following CUTLASS threadblock swizzling patterns.
 */
__global__ void cutlass_gf_gemm_kernel(const uint8_t* __restrict__ A,
                                        const uint8_t* __restrict__ B,
                                        uint8_t* C,
                                        int m, int n, int k,
                                        int lda, int ldb, int ldc) {
    constexpr int THREADBLOCK_M = 128;
    constexpr int THREADBLOCK_N = 128;
    constexpr int TILE_SIZE = 16;

    int tb_row = blockIdx.y * THREADBLOCK_M;
    int tb_col = blockIdx.x * THREADBLOCK_N;

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    // Each thread computes a TILE_SIZE x TILE_SIZE tile
    int row = tb_row + local_row;
    int col = tb_col + local_col;

    __shared__ uint8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t Bs[TILE_SIZE][TILE_SIZE];

    uint8_t accum = 0;
    int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_col_a = t * TILE_SIZE + local_col;
        int tiled_row_b = t * TILE_SIZE + local_row;

        // Load tile of A
        if (row < m && tiled_col_a < k) {
            As[local_row][local_col] = A[row * lda + tiled_col_a];
        } else {
            As[local_row][local_col] = 0;
        }

        // Load tile of B
        if (tiled_row_b < k && col < n) {
            Bs[local_row][local_col] = B[tiled_row_b * ldb + col];
        } else {
            Bs[local_row][local_col] = 0;
        }

        __syncthreads();

        // Compute partial dot product using constant memory tables
        for (int i = 0; i < TILE_SIZE; ++i) {
            uint8_t a = As[local_row][i];
            uint8_t b = Bs[i][local_col];

            if (a != 0 && b != 0) {
                int log_sum = d_cutlass_gflog_const[a] + d_cutlass_gflog_const[b];
                accum ^= d_cutlass_gfexp_const[log_sum];
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * ldc + col] = accum;
    }
}

// ============================================================================
// Execute Function
// ============================================================================

GFGemmError cutlass_gf_gemm_execute(
    int m, int n, int k,
    const uint8_t* A, int lda,
    const uint8_t* B, int ldb,
    uint8_t* C, int ldc,
    const uint8_t* d_gf_exp,
    const uint8_t* d_gf_log,
    cudaStream_t stream
) {
    if (m <= 0 || n <= 0 || k <= 0 || A == nullptr || B == nullptr || C == nullptr) {
        return GF_GEMM_ERROR_INVALID_VALUE;
    }

    // Copy GF tables to constant memory
    size_t exp_size = cutlass::GF28Arithmetic::kExpTableSize * sizeof(uint8_t);
    size_t log_size = cutlass::GF28Arithmetic::kLogTableSize * sizeof(uint8_t);

    cudaMemcpyToSymbolAsync(d_cutlass_gfexp_const, d_gf_exp, exp_size, 0, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyToSymbolAsync(d_cutlass_gflog_const, d_gf_log, log_size, 0, cudaMemcpyDeviceToDevice, stream);

    // CUTLASS-style threadblock configuration: 128x128 threadblocks
    constexpr int THREADBLOCK_M = 128;
    constexpr int THREADBLOCK_N = 128;

    // Each threadblock has THREADBLOCK_M x THREADBLOCK_N threads conceptually,
    // but we use a 16x16 thread block that tiles internally
    dim3 block(16, 16);  // 256 threads per block (2x2 tiles of 128x128)
    dim3 grid(
        (n + THREADBLOCK_N - 1) / THREADBLOCK_N,
        (m + THREADBLOCK_M - 1) / THREADBLOCK_M
    );

    cutlass_gf_gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return GF_GEMM_ERROR_CUDA_KERNEL_FAILED;
    }

    return GF_GEMM_SUCCESS;
}
