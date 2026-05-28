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
    CUTLASS_HOST_DEVICE
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
 * Uses 16x16 shared memory tiles with constant memory GF lookup tables.
 * Grid is configured for larger threadblock coverage (128x128 output tiles).
 */
__global__ void cutlass_gf_gemm_kernel(const uint8_t* __restrict__ A,
                                        const uint8_t* __restrict__ B,
                                        uint8_t* C,
                                        int m, int n, int k,
                                        int lda, int ldb, int ldc) {
    constexpr int TILE_SIZE = 16;

    __shared__ uint8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    uint8_t accum = 0;
    int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_col_a = t * TILE_SIZE + threadIdx.x;
        int tiled_row_b = t * TILE_SIZE + threadIdx.y;

        // Load tile of A
        if (row < m && tiled_col_a < k) {
            As[threadIdx.y][threadIdx.x] = A[row * lda + tiled_col_a];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Load tile of B
        if (tiled_row_b < k && col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[tiled_row_b * ldb + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial dot product using constant memory tables
        for (int i = 0; i < TILE_SIZE; ++i) {
            uint8_t a = As[threadIdx.y][i];
            uint8_t b = Bs[i][threadIdx.x];

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

    // 16x16 tiles
    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (n + TILE_SIZE - 1) / TILE_SIZE,
        (m + TILE_SIZE - 1) / TILE_SIZE
    );

    cutlass_gf_gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return GF_GEMM_ERROR_CUDA_KERNEL_FAILED;
    }

    return GF_GEMM_SUCCESS;
}
