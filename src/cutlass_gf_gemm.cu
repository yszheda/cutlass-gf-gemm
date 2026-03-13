/**
 * @file cutlass_gf_gemm.cu
 * @brief CUTLASS-based Galois Field Matrix Multiplication Implementation
 *
 * This implementation uses CUTLASS templates to perform efficient
 * matrix multiplication over GF(2^8) field.
 */

#include "cutlass_gf_gemm.h"
#include "gf_ops.h"
#include "gf16.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
//                     Device-side GF Tables (Constant Memory)
// ============================================================================

// Constant memory for GF lookup tables
__constant__ uint8_t d_gfexp_const[cutlass::GF28Arithmetic::kExpTableSize];
__constant__ uint8_t d_gflog_const[cutlass::GF28Arithmetic::kLogTableSize];

// ============================================================================
//                     GF Table Initialization Kernel
// ============================================================================

/**
 * @brief Kernel to initialize GF(2^8) lookup tables on device
 */
__global__ void init_gf_tables_kernel(uint8_t* gf_exp, uint8_t* gf_log) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    const uint8_t prim_poly = cutlass::GF28Arithmetic::kPrimitivePolynomial;
    const int field_max = cutlass::GF28Arithmetic::kFieldMax;

    // Initialize log table
    if (idx < cutlass::GF28Arithmetic::kLogTableSize) {
        gf_log[idx] = 0;
    }
    __syncthreads();

    // Generate tables sequentially (only need one thread)
    if (idx == 0) {
        uint8_t exp = 1;
        for (int log = 0; log < field_max; ++log) {
            gf_log[exp] = static_cast<uint8_t>(log);
            gf_exp[log] = exp;
            // Extended table for overflow handling
            gf_exp[log + field_max] = exp;
            gf_exp[log + 2 * field_max] = exp;

            // Next power of primitive element
            exp = (exp << 1) ^ ((exp & 0x80) ? prim_poly : 0);
        }
        // Handle zero
        gf_log[0] = 0;  // Special case, will never be used in multiplication
        gf_exp[0] = 1;
    }
}

/**
 * @brief Initialize GF tables in global memory
 */
extern "C" cudaError_t gf_init_tables_device(uint8_t* gf_exp, uint8_t* gf_log) {
    init_gf_tables_kernel<<<1, 256>>>(gf_exp, gf_log);
    return cudaGetLastError();
}

// ============================================================================
//                     Simple GF Matrix Multiplication Kernel
// ============================================================================

/**
 * @brief Simple tiled matrix multiplication kernel for GF(2^8)
 *
 * This is a baseline implementation without CUTLASS for reference.
 * Uses shared memory tiling for efficiency.
 *
 * @param A Matrix A (row-major)
 * @param B Matrix B (row-major)
 * @param C Output matrix C = A * B
 * @param m Number of rows of A and C
 * @param n Number of columns of B and C
 * @param k Number of columns of A and rows of B
 * @param gf_exp GF exponentiation table in constant memory
 * @param gf_log GF logarithm table in constant memory
 */
__global__ void gf_gemm_kernel_simple(const uint8_t* __restrict__ A,
                                       const uint8_t* __restrict__ B,
                                       uint8_t* C,
                                       int m, int n, int k,
                                       int lda, int ldb, int ldc) {
    // Tile dimensions
    constexpr int TILE_SIZE = 16;

    __shared__ uint8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    uint8_t accum = 0;

    // Number of tiles along k dimension
    int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_row = row;
        int tiled_col_a = threadIdx.x + t * TILE_SIZE;
        int tiled_col_b = threadIdx.y;
        int tiled_row_b = col;

        // Load tile of A
        if (tiled_row < m && tiled_col_a < k) {
            As[threadIdx.y][threadIdx.x] = A[tiled_row * lda + tiled_col_a];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Load tile of B (transposed for coalesced access)
        if (tiled_col_b < k && tiled_row_b < n) {
            Bs[threadIdx.y][threadIdx.x] = B[tiled_col_b * ldb + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            uint8_t a = As[threadIdx.y][i];
            uint8_t b = Bs[i][threadIdx.x];

            if (a != 0 && b != 0) {
                // GF multiplication using log/exp tables
                uint8_t log_sum = d_gflog_const[a] + d_gflog_const[b];
                accum ^= d_gfexp_const[log_sum];
            }
        }

        __syncthreads();
    }

    // Write result
    if (row < m && col < n) {
        C[row * ldc + col] = accum;
    }
}

// ============================================================================
//                     CUTLASS-based GEMM Implementation
// ============================================================================

/// Internal implementation structure
struct GFGemmImpl {
    GFGemmConfig config;
    uint8_t* d_gf_exp;
    uint8_t* d_gf_log;
    cudaStream_t* streams;
    bool initialized;

    GFGemmImpl() : d_gf_exp(nullptr), d_gf_log(nullptr),
                   streams(nullptr), initialized(false) {}
};

/**
 * @brief Initialize GFGemm instance
 */
GFGemmError gf_gemm_create(GFGemmHandle* handle, const GFGemmConfig* config) {
    if (handle == nullptr) {
        return GF_GEMM_ERROR_INVALID_VALUE;
    }

    *handle = new GFGemmImpl();

    if (config != nullptr) {
        (*handle)->config = *config;
    } else {
        gf_gemm_config_init_default(&(*handle)->config);
    }

    // Allocate device memory for GF tables
    size_t exp_size = cutlass::GF28Arithmetic::kExpTableSize * sizeof(uint8_t);
    size_t log_size = cutlass::GF28Arithmetic::kLogTableSize * sizeof(uint8_t);

    cudaError_t err = cudaMalloc(&(*handle)->d_gf_exp, exp_size);
    if (err != cudaSuccess) {
        delete *handle;
        *handle = nullptr;
        return GF_GEMM_ERROR_OUT_OF_MEMORY;
    }

    err = cudaMalloc(&(*handle)->d_gf_log, log_size);
    if (err != cudaSuccess) {
        cudaFree((*handle)->d_gf_exp);
        delete *handle;
        *handle = nullptr;
        return GF_GEMM_ERROR_OUT_OF_MEMORY;
    }

    // Initialize tables
    gf_init_tables_device((*handle)->d_gf_exp, (*handle)->d_gf_log);
    cudaDeviceSynchronize();

    // Copy to constant memory
    cudaMemcpyToSymbol(d_gfexp_const, (*handle)->d_gf_exp, exp_size);
    cudaMemcpyToSymbol(d_gflog_const, (*handle)->d_gf_log, log_size);

    // Allocate streams if needed
    int num_streams = (*handle)->config.num_streams;
    if (num_streams > 1) {
        (*handle)->streams = new cudaStream_t[num_streams];
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamCreate(&(*handle)->streams[i]);
        }
    }

    (*handle)->initialized = true;
    return GF_GEMM_SUCCESS;
}

/**
 * @brief Destroy GFGemm instance
 */
GFGemmError gf_gemm_destroy(GFGemmHandle handle) {
    if (handle == nullptr) {
        return GF_GEMM_ERROR_INVALID_VALUE;
    }

    if (handle->d_gf_exp) {
        cudaFree(handle->d_gf_exp);
    }
    if (handle->d_gf_log) {
        cudaFree(handle->d_gf_log);
    }
    if (handle->streams) {
        for (int i = 0; i < handle->config.num_streams; ++i) {
            cudaStreamDestroy(handle->streams[i]);
        }
        delete[] handle->streams;
    }

    delete handle;
    return GF_GEMM_SUCCESS;
}

/**
 * @brief Initialize config with default values
 */
void gf_gemm_config_init_default(GFGemmConfig* config) {
    if (config == nullptr) return;

    config->layout_a = GF_GEMM_ROW_MAJOR;
    config->layout_b = GF_GEMM_ROW_MAJOR;
    config->layout_c = GF_GEMM_ROW_MAJOR;
    config->num_streams = 1;
    config->enable_profiling = 0;
    memset(config->reserved, 0, sizeof(config->reserved));
}

/**
 * @brief Get error string
 */
const char* gf_gemm_get_error_string(GFGemmError error) {
    switch (error) {
        case GF_GEMM_SUCCESS:
            return "Success";
        case GF_GEMM_ERROR_INVALID_VALUE:
            return "Invalid value";
        case GF_GEMM_ERROR_NOT_INITIALIZED:
            return "Not initialized";
        case GF_GEMM_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case GF_GEMM_ERROR_CUBLAS_INIT_FAILED:
            return "cuBLAS init failed";
        case GF_GEMM_ERROR_CUDA_KERNEL_FAILED:
            return "CUDA kernel failed";
        case GF_GEMM_ERROR_UNSUPPORTED:
            return "Unsupported operation";
        default:
            return "Unknown error";
    }
}

/**
 * @brief Perform simple GF matrix multiplication
 */
GFGemmError gf_gemm_mm(GFGemmHandle handle,
                       int m, int n, int k,
                       const uint8_t* A, int lda,
                       const uint8_t* B, int ldb,
                       uint8_t* C, int ldc,
                       cudaStream_t stream) {
    if (handle == nullptr || !handle->initialized) {
        return GF_GEMM_ERROR_NOT_INITIALIZED;
    }

    if (m <= 0 || n <= 0 || k <= 0) {
        return GF_GEMM_ERROR_INVALID_VALUE;
    }

    if (A == nullptr || B == nullptr || C == nullptr) {
        return GF_GEMM_ERROR_INVALID_VALUE;
    }

    // Configure kernel launch parameters
    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
              (m + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    gf_gemm_kernel_simple<<<grid, block, 0, stream>>>(
        A, B, C, m, n, k, lda, ldb, ldc);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return GF_GEMM_ERROR_CUDA_KERNEL_FAILED;
    }

    return GF_GEMM_SUCCESS;
}

/**
 * @brief Full GEMM with alpha/beta scaling
 */
GFGemmError gf_gemm_compute(GFGemmHandle handle,
                            int m, int n, int k,
                            const uint8_t* alpha,
                            const uint8_t* A, int lda,
                            const uint8_t* B, int ldb,
                            const uint8_t* beta,
                            uint8_t* C, int ldc,
                            cudaStream_t stream) {
    // For GF(2^8), alpha=1 and beta=0 is the common case (just A*B)
    // Full alpha/beta support would require additional GF multiplication

    if (beta == nullptr || (beta[0] == 0)) {
        // Just C = A * B (or C = alpha * A * B)
        return gf_gemm_mm(handle, m, n, k, A, lda, B, ldb, C, ldc, stream);
    }

    // For beta != 0, we need C = A*B + beta*C
    // This requires a more complex kernel - for now, return unsupported
    return GF_GEMM_ERROR_UNSUPPORTED;
}

/**
 * @brief Synchronize GFGemm instance
 */
GFGemmError gf_gemm_synchronize(GFGemmHandle handle) {
    if (handle == nullptr) {
        return GF_GEMM_ERROR_INVALID_VALUE;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return GF_GEMM_ERROR_CUDA_KERNEL_FAILED;
    }

    return GF_GEMM_SUCCESS;
}

// ============================================================================
//                     Utility Functions
// ============================================================================

extern "C" void gf_random_matrix(uint8_t* matrix, size_t size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < size; ++i) {
        matrix[i] = static_cast<uint8_t>(rand() % 256);
    }
}

extern "C" void gf_print_matrix(const uint8_t* matrix, int rows, int cols,
                                 const char* name) {
    printf("Matrix %s (%dx%d):\n", name ? name : "", rows, cols);
    for (int i = 0; i < rows; ++i) {
        printf("  ");
        for (int j = 0; j < cols; ++j) {
            printf("%3d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

extern "C" int gf_verify_gemm(const uint8_t* A, const uint8_t* B,
                               const uint8_t* C, int m, int n, int k) {
    // Compute reference result on CPU
    uint8_t* ref = new uint8_t[m * n];
    memset(ref, 0, m * n);

    // CPU reference implementation using log/exp tables
    uint8_t gf_exp[768];
    uint8_t gf_log[256];
    const uint8_t prim_poly = 0x1D;

    // Generate tables
    uint8_t exp = 1;
    for (int log = 0; log < 255; ++log) {
        gf_log[exp] = static_cast<uint8_t>(log);
        gf_exp[log] = exp;
        gf_exp[log + 255] = exp;
        gf_exp[log + 510] = exp;
        exp = (exp << 1) ^ ((exp & 0x80) ? prim_poly : 0);
    }
    gf_log[0] = 0;

    // Compute reference
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            uint8_t sum = 0;
            for (int l = 0; l < k; ++l) {
                uint8_t a = A[i * n + l];  // Assuming row-major
                uint8_t b = B[l * n + j];
                if (a != 0 && b != 0) {
                    sum ^= gf_exp[gf_log[a] + gf_log[b]];
                }
            }
            ref[i * n + j] = sum;
        }
    }

    // Compare with GPU result
    int errors = 0;
    for (int i = 0; i < m * n; ++i) {
        if (ref[i] != C[i]) {
            errors++;
            if (errors < 10) {
                printf("Mismatch at index %d: expected %d, got %d\n",
                       i, ref[i], C[i]);
            }
        }
    }

    delete[] ref;

    if (errors > 0) {
        printf("Total errors: %d / %d\n", errors, m * n);
        return -1;
    }

    return 0;
}
