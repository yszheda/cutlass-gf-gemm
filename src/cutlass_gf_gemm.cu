/**
 * @file cutlass_gf_gemm.cu
 * @brief CUTLASS-based Galois Field Matrix Multiplication Implementation
 */

#include "cutlass_gf_gemm.h"
#include "gf_ops.h"
#include "gf16.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Constant memory for GF lookup tables
__constant__ uint8_t d_gfexp_const[cutlass::GF28Arithmetic::kExpTableSize];
__constant__ uint8_t d_gflog_const[cutlass::GF28Arithmetic::kLogTableSize];

/**
 * @brief Kernel to initialize GF(2^8) lookup tables on device
 */
__global__ void init_gf_tables_kernel(uint8_t* gf_exp, uint8_t* gf_log) {
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

extern "C" cudaError_t gf_init_tables_device(uint8_t* gf_exp, uint8_t* gf_log) {
    init_gf_tables_kernel<<<1, 1>>>(gf_exp, gf_log);
    return cudaGetLastError();
}

/**
 * @brief Simple tiled matrix multiplication kernel for GF(2^8)
 */
__global__ void gf_gemm_kernel_simple(const uint8_t* __restrict__ A,
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

        // Compute partial dot product
        for (int i = 0; i < TILE_SIZE; ++i) {
            uint8_t a = As[threadIdx.y][i];
            uint8_t b = Bs[i][threadIdx.x];

            if (a != 0 && b != 0) {
                // Use int to avoid overflow: log values can sum up to 508 (254+254)
                int log_sum = d_gflog_const[a] + d_gflog_const[b];
                accum ^= d_gfexp_const[log_sum];
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * ldc + col] = accum;
    }
}

/**
 * @brief Tiled matrix multiplication kernel for GF(2^8) with alpha/beta scaling
 *
 * Computes: C = alpha * (A * B) + beta * C
 *
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Input/output matrix C
 * @param alpha Scalar multiplier for A*B (GF element)
 * @param beta Scalar multiplier for original C (GF element)
 * @param m Number of rows of A and C
 * @param n Number of columns of B and C
 * @param k Number of columns of A and rows of B
 * @param lda Leading dimension of A
 * @param ldb Leading dimension of B
 * @param ldc Leading dimension of C
 */
__global__ void gf_gemm_kernel_scaled(const uint8_t* __restrict__ A,
                                       const uint8_t* __restrict__ B,
                                       uint8_t* C,
                                       uint8_t alpha,
                                       uint8_t beta,
                                       int m, int n, int k,
                                       int lda, int ldb, int ldc) {
    constexpr int TILE_SIZE = 16;

    __shared__ uint8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    uint8_t accum = 0;
    int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    // Load and scale original C if beta != 0
    uint8_t c_scaled = 0;
    if (beta != 0 && row < m && col < n) {
        uint8_t c_val = C[row * ldc + col];
        if (beta == 1) {
            c_scaled = c_val;
        } else if (c_val != 0) {
            int log_sum = d_gflog_const[c_val] + d_gflog_const[beta];
            c_scaled = d_gfexp_const[log_sum];
        }
    }

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

        // Compute partial dot product
        for (int i = 0; i < TILE_SIZE; ++i) {
            uint8_t a = As[threadIdx.y][i];
            uint8_t b = Bs[i][threadIdx.x];

            if (a != 0 && b != 0) {
                int log_sum = d_gflog_const[a] + d_gflog_const[b];
                accum ^= d_gfexp_const[log_sum];
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        // Apply alpha scaling to result
        if (alpha == 0) {
            accum = 0;
        } else if (alpha != 1) {
            int log_sum = d_gflog_const[accum] + d_gflog_const[alpha];
            accum = d_gfexp_const[log_sum];
        }
        // Add beta * C (XOR in GF(2^8))
        accum ^= c_scaled;
        C[row * ldc + col] = accum;
    }
}

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

    gf_init_tables_device((*handle)->d_gf_exp, (*handle)->d_gf_log);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree((*handle)->d_gf_exp);
        cudaFree((*handle)->d_gf_log);
        delete *handle;
        *handle = nullptr;
        return GF_GEMM_ERROR_CUDA_KERNEL_FAILED;
    }

    cudaMemcpyToSymbol(d_gfexp_const, (*handle)->d_gf_exp, exp_size);
    cudaMemcpyToSymbol(d_gflog_const, (*handle)->d_gf_log, log_size);

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

GFGemmError gf_gemm_destroy(GFGemmHandle handle) {
    if (handle == nullptr) {
        return GF_GEMM_ERROR_INVALID_VALUE;
    }

    if (handle->d_gf_exp) cudaFree(handle->d_gf_exp);
    if (handle->d_gf_log) cudaFree(handle->d_gf_log);
    if (handle->streams) {
        for (int i = 0; i < handle->config.num_streams; ++i) {
            cudaStreamDestroy(handle->streams[i]);
        }
        delete[] handle->streams;
    }

    delete handle;
    return GF_GEMM_SUCCESS;
}

void gf_gemm_config_init_default(GFGemmConfig* config) {
    if (config == nullptr) return;
    config->layout_a = GF_GEMM_ROW_MAJOR;
    config->layout_b = GF_GEMM_ROW_MAJOR;
    config->layout_c = GF_GEMM_ROW_MAJOR;
    config->num_streams = 1;
    config->enable_profiling = 0;
    memset(config->reserved, 0, sizeof(config->reserved));
}

const char* gf_gemm_get_error_string(GFGemmError error) {
    switch (error) {
        case GF_GEMM_SUCCESS: return "Success";
        case GF_GEMM_ERROR_INVALID_VALUE: return "Invalid value";
        case GF_GEMM_ERROR_NOT_INITIALIZED: return "Not initialized";
        case GF_GEMM_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case GF_GEMM_ERROR_CUBLAS_INIT_FAILED: return "cuBLAS init failed";
        case GF_GEMM_ERROR_CUDA_KERNEL_FAILED: return "CUDA kernel failed";
        case GF_GEMM_ERROR_UNSUPPORTED: return "Unsupported operation";
        default: return "Unknown error";
    }
}

GFGemmError gf_gemm_mm(GFGemmHandle handle,
                       int m, int n, int k,
                       const uint8_t* A, int lda,
                       const uint8_t* B, int ldb,
                       uint8_t* C, int ldc,
                       cudaStream_t stream) {
    if (handle == nullptr || !handle->initialized) {
        return GF_GEMM_ERROR_NOT_INITIALIZED;
    }

    if (m <= 0 || n <= 0 || k <= 0 || A == nullptr || B == nullptr || C == nullptr) {
        return GF_GEMM_ERROR_INVALID_VALUE;
    }

    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
              (m + TILE_SIZE - 1) / TILE_SIZE);

    gf_gemm_kernel_simple<<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return GF_GEMM_ERROR_CUDA_KERNEL_FAILED;
    }

    return GF_GEMM_SUCCESS;
}

GFGemmError gf_gemm_compute(GFGemmHandle handle,
                            int m, int n, int k,
                            const uint8_t* alpha,
                            const uint8_t* A, int lda,
                            const uint8_t* B, int ldb,
                            const uint8_t* beta,
                            uint8_t* C, int ldc,
                            cudaStream_t stream) {
    if (handle == nullptr || !handle->initialized) {
        return GF_GEMM_ERROR_NOT_INITIALIZED;
    }

    if (m <= 0 || n <= 0 || k <= 0 || A == nullptr || B == nullptr || C == nullptr) {
        return GF_GEMM_ERROR_INVALID_VALUE;
    }

    // Handle default values for alpha and beta
    uint8_t alpha_val = (alpha != nullptr) ? alpha[0] : 1;
    uint8_t beta_val = (beta != nullptr) ? beta[0] : 0;

    // Fast path: if alpha=1 and beta=0, use simple kernel
    if (alpha_val == 1 && beta_val == 0) {
        return gf_gemm_mm(handle, m, n, k, A, lda, B, ldb, C, ldc, stream);
    }

    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
              (m + TILE_SIZE - 1) / TILE_SIZE);

    gf_gemm_kernel_scaled<<<grid, block, 0, stream>>>(A, B, C, alpha_val, beta_val,
                                                       m, n, k, lda, ldb, ldc);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return GF_GEMM_ERROR_CUDA_KERNEL_FAILED;
    }

    return GF_GEMM_SUCCESS;
}

GFGemmError gf_gemm_synchronize(GFGemmHandle handle) {
    if (handle == nullptr) return GF_GEMM_ERROR_INVALID_VALUE;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return GF_GEMM_ERROR_CUDA_KERNEL_FAILED;
    return GF_GEMM_SUCCESS;
}

extern "C" void gf_random_matrix(uint8_t* matrix, size_t size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < size; ++i) {
        matrix[i] = static_cast<uint8_t>(rand() % 256);
    }
}

extern "C" void gf_print_matrix(const uint8_t* matrix, int rows, int cols, const char* name) {
    printf("Matrix %s (%dx%d):\n", name ? name : "", rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%3d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

extern "C" int gf_verify_gemm(const uint8_t* A, const uint8_t* B, const uint8_t* C,
                               int m, int n, int k) {
    uint8_t* ref = new uint8_t[m * n];
    memset(ref, 0, m * n);

    uint8_t gf_exp[768];
    uint8_t gf_log[256];
    const uint8_t prim_poly = 0x1D;

    uint8_t exp = 1;
    for (int log = 0; log < 255; ++log) {
        gf_log[exp] = static_cast<uint8_t>(log);
        gf_exp[log] = exp;
        gf_exp[log + 255] = exp;
        gf_exp[log + 510] = exp;
        exp = (exp << 1) ^ ((exp & 0x80) ? prim_poly : 0);
    }
    gf_log[0] = 0;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            uint8_t sum = 0;
            for (int l = 0; l < k; ++l) {
                uint8_t a = A[i * k + l];
                uint8_t b = B[l * n + j];
                if (a != 0 && b != 0) {
                    sum ^= gf_exp[gf_log[a] + gf_log[b]];
                }
            }
            ref[i * n + j] = sum;
        }
    }

    int errors = 0;
    for (int i = 0; i < m * n; ++i) {
        if (ref[i] != C[i]) {
            errors++;
            if (errors < 10) {
                printf("Mismatch at [%d,%d] (index %d): expected %d, got %d\n",
                       i / n, i % n, i, ref[i], C[i]);
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
