/**
 * @file test_gf_gemm.cu
 * @brief Comprehensive tests for CUTLASS GF Gemm library
 */

#include "cutlass_gf_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return 1; \
        } \
    } while (0)

#define GF_GEMM_CHECK(call) \
    do { \
        GFGemmError err = call; \
        if (err != GF_GEMM_SUCCESS) { \
            fprintf(stderr, "GF Gemm error at %s:%d: %s\n", \
                    __FILE__, __LINE__, gf_gemm_get_error_string(err)); \
            return 1; \
        } \
    } while (0)

/**
 * @brief CPU reference GF(2^8) matrix multiplication
 */
void gf_gemm_reference(const uint8_t* A, const uint8_t* B, uint8_t* C,
                       int m, int n, int k) {
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
    gf_exp[0] = 1;

    memset(C, 0, m * n);
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
            C[i * n + j] = sum;
        }
    }
}

int test_basic_gemm() {
    printf("\n=== Test: Basic GEMM ===\n");

    int m = 32, n = 32, k = 32;
    size_t size_a = m * k;
    size_t size_b = k * n;
    size_t size_c = m * n;

    uint8_t* h_A = (uint8_t*)malloc(size_a);
    uint8_t* h_B = (uint8_t*)malloc(size_b);
    uint8_t* h_C_gpu = (uint8_t*)malloc(size_c);
    uint8_t* h_C_cpu = (uint8_t*)malloc(size_c);

    for (size_t i = 0; i < size_a; ++i) h_A[i] = i % 256;
    for (size_t i = 0; i < size_b; ++i) h_B[i] = (i * 7) % 256;

    uint8_t *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_a));
    CUDA_CHECK(cudaMalloc(&d_B, size_b));
    CUDA_CHECK(cudaMalloc(&d_C, size_c));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    GFGemmHandle handle;
    GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));
    GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));
    gf_gemm_reference(h_A, h_B, h_C_cpu, m, n, k);

    int errors = 0;
    for (size_t i = 0; i < size_c; ++i) {
        if (h_C_gpu[i] != h_C_cpu[i]) errors++;
    }

    printf("  Matrix size: %dx%d * %dx%d = %dx%d\n", m, k, k, n, m, n);
    printf("  Errors: %d / %zu\n", errors, size_c);

    gf_gemm_destroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);

    return errors > 0 ? 1 : 0;
}

int test_identity() {
    printf("\n=== Test: Identity Matrix ===\n");

    int m = 16, n = 16, k = 16;
    size_t size_a = m * k, size_b = k * n, size_c = m * n;

    uint8_t* h_A = (uint8_t*)calloc(size_a, 1);
    uint8_t* h_B = (uint8_t*)calloc(size_b, 1);
    uint8_t* h_C = (uint8_t*)malloc(size_c);

    for (int i = 0; i < m; ++i) h_A[i * k + i] = 1;
    srand(123);
    for (size_t i = 0; i < size_b; ++i) h_B[i] = rand() % 256;

    uint8_t *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_a));
    CUDA_CHECK(cudaMalloc(&d_B, size_b));
    CUDA_CHECK(cudaMalloc(&d_C, size_c));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    GFGemmHandle handle;
    GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));
    GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_c, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (size_t i = 0; i < size_c; ++i) {
        if (h_C[i] != h_B[i]) errors++;
    }

    printf("  Identity * Random = Random\n");
    printf("  Errors: %d / %zu\n", errors, size_c);

    gf_gemm_destroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return errors > 0 ? 1 : 0;
}

int test_zero_matrix() {
    printf("\n=== Test: Zero Matrix ===\n");

    int m = 16, n = 16, k = 16;
    size_t size_a = m * k, size_b = k * n, size_c = m * n;

    uint8_t* h_A = (uint8_t*)calloc(size_a, 1);
    uint8_t* h_B = (uint8_t*)calloc(size_b, 1);
    uint8_t* h_C = (uint8_t*)malloc(size_c);

    srand(456);
    for (size_t i = 0; i < size_b; ++i) h_B[i] = rand() % 256;

    uint8_t *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_a));
    CUDA_CHECK(cudaMalloc(&d_B, size_b));
    CUDA_CHECK(cudaMalloc(&d_C, size_c));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    GFGemmHandle handle;
    GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));
    GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_c, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (size_t i = 0; i < size_c; ++i) {
        if (h_C[i] != 0) errors++;
    }

    printf("  Zero * Random = Zero\n");
    printf("  Errors: %d / %zu\n", errors, size_c);

    gf_gemm_destroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return errors > 0 ? 1 : 0;
}

int test_various_sizes() {
    printf("\n=== Test: Various Sizes ===\n");

    struct TestCase { int m, n, k; const char* name; };
    TestCase test_cases[] = {
        {4, 4, 4, "4x4 * 4x4"},
        {8, 16, 8, "8x8 * 8x16"},
        {16, 32, 16, "16x16 * 16x32"},
        {32, 64, 32, "32x32 * 32x64"},
        {64, 128, 64, "64x64 * 64x128"},
        {128, 256, 128, "128x128 * 128x256"},
        {256, 512, 256, "256x256 * 256x512"},
    };

    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int total_errors = 0;

    GFGemmHandle handle;
    GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));

    for (int i = 0; i < num_tests; ++i) {
        int m = test_cases[i].m, n = test_cases[i].n, k = test_cases[i].k;
        size_t size_a = m * k, size_b = k * n, size_c = m * n;

        uint8_t* h_A = (uint8_t*)malloc(size_a);
        uint8_t* h_B = (uint8_t*)malloc(size_b);
        uint8_t* h_C_gpu = (uint8_t*)malloc(size_c);
        uint8_t* h_C_cpu = (uint8_t*)malloc(size_c);

        srand(789 + i);
        for (size_t j = 0; j < size_a; ++j) h_A[j] = rand() % 256;
        for (size_t j = 0; j < size_b; ++j) h_B[j] = rand() % 256;

        uint8_t *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_a));
        CUDA_CHECK(cudaMalloc(&d_B, size_b));
        CUDA_CHECK(cudaMalloc(&d_C, size_c));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

        GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));
        gf_gemm_reference(h_A, h_B, h_C_cpu, m, n, k);

        int errors = 0;
        for (size_t j = 0; j < size_c; ++j) {
            if (h_C_gpu[j] != h_C_cpu[j]) errors++;
        }

        printf("  %s: %d errors\n", test_cases[i].name, errors);
        total_errors += errors;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    }

    gf_gemm_destroy(handle);
    printf("  Total errors: %d\n", total_errors);

    return total_errors > 0 ? 1 : 0;
}

int test_performance() {
    printf("\n=== Test: Performance Benchmark ===\n");

    struct BenchmarkCase { int m, n, k; };
    BenchmarkCase cases[] = {
        {64, 64, 64}, {128, 128, 128}, {256, 256, 256},
        {512, 512, 512}, {1024, 1024, 1024},
    };

    int num_cases = sizeof(cases) / sizeof(cases[0]);
    GFGemmHandle handle;
    GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < num_cases; ++i) {
        int m = cases[i].m, n = cases[i].n, k = cases[i].k;
        size_t size_a = m * k, size_b = k * n, size_c = m * n;

        uint8_t *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_a));
        CUDA_CHECK(cudaMalloc(&d_B, size_b));
        CUDA_CHECK(cudaMalloc(&d_C, size_c));

        GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        int num_iters = 10;
        cudaEventRecord(start, 0);
        for (int j = 0; j < num_iters; ++j) {
            GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));
        }
        cudaEventRecord(stop, 0);
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        float avg_ms = elapsed_ms / num_iters;
        float gmacs = (2.0f * m * n * k) / (avg_ms * 1e6f);

        printf("  %4d x %4d x %4d: %8.3f ms, %8.2f GMACS\n", m, n, k, avg_ms, gmacs);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    gf_gemm_destroy(handle);

    return 0;
}

/**
 * @brief Test alpha/beta scaling in gf_gemm_compute
 */
int test_alpha_beta_scaling() {
    printf("\n=== Test: Alpha/Beta Scaling ===\n");

    int m = 16, n = 16, k = 16;
    size_t size_a = m * k, size_b = k * n, size_c = m * n;

    uint8_t* h_A = (uint8_t*)malloc(size_a);
    uint8_t* h_B = (uint8_t*)malloc(size_b);
    uint8_t* h_C = (uint8_t*)malloc(size_c);
    uint8_t* h_C_gpu = (uint8_t*)malloc(size_c);
    uint8_t* h_C_cpu = (uint8_t*)malloc(size_c);

    // Fill matrices with known values
    for (size_t i = 0; i < size_a; ++i) h_A[i] = (i + 1) % 256;
    for (size_t i = 0; i < size_b; ++i) h_B[i] = (i * 3 + 7) % 256;
    for (size_t i = 0; i < size_c; ++i) h_C[i] = (i * 5 + 11) % 256;

    uint8_t *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_a));
    CUDA_CHECK(cudaMalloc(&d_B, size_b));
    CUDA_CHECK(cudaMalloc(&d_C, size_c));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_c, cudaMemcpyHostToDevice));

    GFGemmHandle handle;
    GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));

    // Build GF tables for CPU reference
    uint8_t gf_exp[768], gf_log[256];
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
    gf_exp[0] = 1;

    int total_errors = 0;

    // Test case 1: alpha=1, beta=0 (should equal simple mm)
    {
        uint8_t alpha = 1, beta = 0;
        memset(h_C_cpu, 0, size_c);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                uint8_t sum = 0;
                for (int l = 0; l < k; ++l) {
                    uint8_t a = h_A[i * k + l];
                    uint8_t b = h_B[l * n + j];
                    if (a != 0 && b != 0) {
                        sum ^= gf_exp[gf_log[a] + gf_log[b]];
                    }
                }
                h_C_cpu[i * n + j] = sum;
            }
        }

        GF_GEMM_CHECK(gf_gemm_compute(handle, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, 0));
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

        int errors = 0;
        for (size_t i = 0; i < size_c; ++i) {
            if (h_C_gpu[i] != h_C_cpu[i]) errors++;
        }
        printf("  alpha=1, beta=0: %d errors\n", errors);
        total_errors += errors;
    }

    // Test case 2: alpha=0, beta=0 (result should be all zeros)
    {
        uint8_t alpha = 0, beta = 0;
        CUDA_CHECK(cudaMemcpy(d_C, h_C, size_c, cudaMemcpyHostToDevice)); // restore C

        GF_GEMM_CHECK(gf_gemm_compute(handle, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, 0));
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

        int errors = 0;
        for (size_t i = 0; i < size_c; ++i) {
            if (h_C_gpu[i] != 0) errors++;
        }
        printf("  alpha=0, beta=0: %d errors\n", errors);
        total_errors += errors;
    }

    // Test case 3: alpha=1, beta=1 (result = A*B + C)
    {
        uint8_t alpha = 1, beta = 1;
        CUDA_CHECK(cudaMemcpy(d_C, h_C, size_c, cudaMemcpyHostToDevice));

        // CPU reference: C_out = A*B + C_in (XOR in GF)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                uint8_t sum = 0;
                for (int l = 0; l < k; ++l) {
                    uint8_t a = h_A[i * k + l];
                    uint8_t b = h_B[l * n + j];
                    if (a != 0 && b != 0) {
                        sum ^= gf_exp[gf_log[a] + gf_log[b]];
                    }
                }
                h_C_cpu[i * n + j] = sum ^ h_C[i * n + j]; // beta*C = C when beta=1
            }
        }

        GF_GEMM_CHECK(gf_gemm_compute(handle, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, 0));
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

        int errors = 0;
        for (size_t i = 0; i < size_c; ++i) {
            if (h_C_gpu[i] != h_C_cpu[i]) errors++;
        }
        printf("  alpha=1, beta=1: %d errors\n", errors);
        total_errors += errors;
    }

    // Test case 4: alpha=2, beta=0 (scale result by 2)
    {
        uint8_t alpha = 2, beta = 0;
        CUDA_CHECK(cudaMemcpy(d_C, h_C, size_c, cudaMemcpyHostToDevice));

        // CPU reference: C_out = 2 * (A*B)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                uint8_t sum = 0;
                for (int l = 0; l < k; ++l) {
                    uint8_t a = h_A[i * k + l];
                    uint8_t b = h_B[l * n + j];
                    if (a != 0 && b != 0) {
                        sum ^= gf_exp[gf_log[a] + gf_log[b]];
                    }
                }
                // Apply alpha=2 scaling
                if (sum != 0) {
                    sum = gf_exp[gf_log[sum] + gf_log[2]];
                }
                h_C_cpu[i * n + j] = sum;
            }
        }

        GF_GEMM_CHECK(gf_gemm_compute(handle, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, 0));
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

        int errors = 0;
        for (size_t i = 0; i < size_c; ++i) {
            if (h_C_gpu[i] != h_C_cpu[i]) errors++;
        }
        printf("  alpha=2, beta=0: %d errors\n", errors);
        total_errors += errors;
    }

    // Test case 5: alpha=1, beta=2 (result = A*B + 2*C)
    {
        uint8_t alpha = 1, beta = 2;
        CUDA_CHECK(cudaMemcpy(d_C, h_C, size_c, cudaMemcpyHostToDevice));

        // CPU reference: C_out = A*B + 2*C
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                uint8_t sum = 0;
                for (int l = 0; l < k; ++l) {
                    uint8_t a = h_A[i * k + l];
                    uint8_t b = h_B[l * n + j];
                    if (a != 0 && b != 0) {
                        sum ^= gf_exp[gf_log[a] + gf_log[b]];
                    }
                }
                // Add beta*C = 2*C
                uint8_t c_scaled = h_C[i * n + j];
                if (c_scaled != 0) {
                    c_scaled = gf_exp[gf_log[c_scaled] + gf_log[2]];
                }
                h_C_cpu[i * n + j] = sum ^ c_scaled;
            }
        }

        GF_GEMM_CHECK(gf_gemm_compute(handle, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, 0));
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

        int errors = 0;
        for (size_t i = 0; i < size_c; ++i) {
            if (h_C_gpu[i] != h_C_cpu[i]) errors++;
        }
        printf("  alpha=1, beta=2: %d errors\n", errors);
        total_errors += errors;
    }

    printf("  Total errors: %d\n", total_errors);

    gf_gemm_destroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_gpu); free(h_C_cpu);

    return total_errors > 0 ? 1 : 0;
}

/**
 * @brief Test minimum size (1x1) matrices
 */
int test_minimum_size() {
    printf("\n=== Test: Minimum Size (1x1) ===\n");

    int m = 1, n = 1, k = 1;

    uint8_t h_A[1] = {100};
    uint8_t h_B[1] = {50};
    uint8_t h_C_gpu[1], h_C_cpu[1];

    // CPU reference: 100 * 50 in GF(2^8)
    const uint8_t prim_poly = 0x1D;
    uint8_t gf_exp[768], gf_log[256];
    uint8_t exp = 1;
    for (int log = 0; log < 255; ++log) {
        gf_log[exp] = static_cast<uint8_t>(log);
        gf_exp[log] = exp;
        gf_exp[log + 255] = exp;
        gf_exp[log + 510] = exp;
        exp = (exp << 1) ^ ((exp & 0x80) ? prim_poly : 0);
    }
    gf_log[0] = 0;
    gf_exp[0] = 1;

    h_C_cpu[0] = (h_A[0] != 0 && h_B[0] != 0) ? gf_exp[gf_log[h_A[0]] + gf_log[h_B[0]]] : 0;

    uint8_t *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, 1));
    CUDA_CHECK(cudaMalloc(&d_B, 1));
    CUDA_CHECK(cudaMalloc(&d_C, 1));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, 1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, 1, cudaMemcpyHostToDevice));

    GFGemmHandle handle;
    GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));
    GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, 1, cudaMemcpyDeviceToHost));

    int errors = (h_C_gpu[0] != h_C_cpu[0]) ? 1 : 0;
    printf("  1x1 * 1x1 = 1x1: expected %d, got %d, errors: %d\n", h_C_cpu[0], h_C_gpu[0], errors);

    gf_gemm_destroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return errors;
}

/**
 * @brief Test non-16-aligned sizes (tile boundary cases)
 */
int test_non_aligned_sizes() {
    printf("\n=== Test: Non-16-Aligned Sizes ===\n");

    struct TestCase { int m, n, k; const char* name; };
    TestCase test_cases[] = {
        {1, 1, 1, "1x1x1"},
        {15, 15, 15, "15x15x15 (just under tile)"},
        {17, 17, 17, "17x17x17 (just over tile)"},
        {31, 31, 31, "31x31x31 (just under 2 tiles)"},
        {33, 33, 33, "33x33x33 (just over 2 tiles)"},
        {17, 31, 19, "17x31x19 (mixed non-aligned)"},
        {65, 127, 63, "65x127x63 (large non-aligned)"},
    };

    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int total_errors = 0;

    // Build GF tables
    uint8_t gf_exp[768], gf_log[256];
    const uint8_t prim_poly = 0x1D;
    uint8_t exp_val = 1;
    for (int log = 0; log < 255; ++log) {
        gf_log[exp_val] = static_cast<uint8_t>(log);
        gf_exp[log] = exp_val;
        gf_exp[log + 255] = exp_val;
        gf_exp[log + 510] = exp_val;
        exp_val = (exp_val << 1) ^ ((exp_val & 0x80) ? prim_poly : 0);
    }
    gf_log[0] = 0;
    gf_exp[0] = 1;

    GFGemmHandle handle;
    GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));

    for (int i = 0; i < num_tests; ++i) {
        int m = test_cases[i].m, n = test_cases[i].n, k = test_cases[i].k;
        size_t size_a = m * k, size_b = k * n, size_c = m * n;

        uint8_t* h_A = (uint8_t*)malloc(size_a);
        uint8_t* h_B = (uint8_t*)malloc(size_b);
        uint8_t* h_C_gpu = (uint8_t*)malloc(size_c);
        uint8_t* h_C_cpu = (uint8_t*)malloc(size_c);

        srand(1234 + i);
        for (size_t j = 0; j < size_a; ++j) h_A[j] = rand() % 256;
        for (size_t j = 0; j < size_b; ++j) h_B[j] = rand() % 256;

        uint8_t *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_a));
        CUDA_CHECK(cudaMalloc(&d_B, size_b));
        CUDA_CHECK(cudaMalloc(&d_C, size_c));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

        GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

        // CPU reference
        memset(h_C_cpu, 0, size_c);
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                uint8_t sum = 0;
                for (int l = 0; l < k; ++l) {
                    uint8_t a = h_A[row * k + l];
                    uint8_t b = h_B[l * n + col];
                    if (a != 0 && b != 0) {
                        sum ^= gf_exp[gf_log[a] + gf_log[b]];
                    }
                }
                h_C_cpu[row * n + col] = sum;
            }
        }

        int errors = 0;
        for (size_t j = 0; j < size_c; ++j) {
            if (h_C_gpu[j] != h_C_cpu[j]) errors++;
        }

        printf("  %s: %d errors\n", test_cases[i].name, errors);
        total_errors += errors;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    }

    gf_gemm_destroy(handle);
    printf("  Total errors: %d\n", total_errors);

    return total_errors > 0 ? 1 : 0;
}

/**
 * @brief Test GF(2^8) boundary values (0, 1, 255)
 */
int test_gf_boundary_values() {
    printf("\n=== Test: GF(2^8) Boundary Values ===\n");

    int m = 8, n = 8, k = 8;
    size_t size_a = m * k, size_b = k * n, size_c = m * n;

    // Build GF tables
    uint8_t gf_exp[768], gf_log[256];
    const uint8_t prim_poly = 0x1D;
    uint8_t exp_val = 1;
    for (int log = 0; log < 255; ++log) {
        gf_log[exp_val] = static_cast<uint8_t>(log);
        gf_exp[log] = exp_val;
        gf_exp[log + 255] = exp_val;
        gf_exp[log + 510] = exp_val;
        exp_val = (exp_val << 1) ^ ((exp_val & 0x80) ? prim_poly : 0);
    }
    gf_log[0] = 0;
    gf_exp[0] = 1;

    int total_errors = 0;

    // Test 1: All zeros
    {
        uint8_t* h_A = (uint8_t*)calloc(size_a, 1);
        uint8_t* h_B = (uint8_t*)calloc(size_b, 1);
        uint8_t* h_C_gpu = (uint8_t*)malloc(size_c);
        uint8_t* h_C_cpu = (uint8_t*)calloc(size_c, 1);

        uint8_t *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_a));
        CUDA_CHECK(cudaMalloc(&d_B, size_b));
        CUDA_CHECK(cudaMalloc(&d_C, size_c));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

        GFGemmHandle handle;
        GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));
        GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));

        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

        int errors = 0;
        for (size_t i = 0; i < size_c; ++i) {
            if (h_C_gpu[i] != h_C_cpu[i]) errors++;
        }
        printf("  All zeros: %d errors\n", errors);
        total_errors += errors;

        gf_gemm_destroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    }

    // Test 2: All ones (1 * 1 = 1 in GF, since log(1)=0, exp(0+0)=1)
    {
        uint8_t* h_A = (uint8_t*)malloc(size_a);
        uint8_t* h_B = (uint8_t*)malloc(size_b);
        uint8_t* h_C_gpu = (uint8_t*)malloc(size_c);
        uint8_t* h_C_cpu = (uint8_t*)malloc(size_c);

        memset(h_A, 1, size_a);
        memset(h_B, 1, size_b);
        // 1 * 1 = exp(log(1) + log(1)) = exp(0 + 0) = exp(0) = 1
        // But we're doing k reductions, XORing 1 k times
        // If k is even, result = 0; if k is odd, result = 1
        memset(h_C_cpu, (k % 2 == 1) ? 1 : 0, size_c);

        uint8_t *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_a));
        CUDA_CHECK(cudaMalloc(&d_B, size_b));
        CUDA_CHECK(cudaMalloc(&d_C, size_c));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

        GFGemmHandle handle;
        GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));
        GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));

        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

        int errors = 0;
        for (size_t i = 0; i < size_c; ++i) {
            if (h_C_gpu[i] != h_C_cpu[i]) errors++;
        }
        printf("  All ones (k=%d): %d errors\n", k, errors);
        total_errors += errors;

        gf_gemm_destroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    }

    // Test 3: Maximum values (255)
    {
        uint8_t* h_A = (uint8_t*)malloc(size_a);
        uint8_t* h_B = (uint8_t*)malloc(size_b);
        uint8_t* h_C_gpu = (uint8_t*)malloc(size_c);
        uint8_t* h_C_cpu = (uint8_t*)malloc(size_c);

        memset(h_A, 255, size_a);
        memset(h_B, 255, size_b);

        // CPU reference
        memset(h_C_cpu, 0, size_c);
        for (int i = 0; i < m * n; ++i) {
            for (int l = 0; l < k; ++l) {
                uint8_t a = 255, b = 255;
                if (a != 0 && b != 0) {
                    h_C_cpu[i] ^= gf_exp[gf_log[a] + gf_log[b]];
                }
            }
        }

        uint8_t *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_a));
        CUDA_CHECK(cudaMalloc(&d_B, size_b));
        CUDA_CHECK(cudaMalloc(&d_C, size_c));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

        GFGemmHandle handle;
        GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));
        GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));

        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

        int errors = 0;
        for (size_t i = 0; i < size_c; ++i) {
            if (h_C_gpu[i] != h_C_cpu[i]) errors++;
        }
        printf("  All 255: %d errors\n", errors);
        total_errors += errors;

        gf_gemm_destroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    }

    // Test 4: Mixed boundary (0, 1, 255)
    {
        uint8_t* h_A = (uint8_t*)malloc(size_a);
        uint8_t* h_B = (uint8_t*)malloc(size_b);
        uint8_t* h_C_gpu = (uint8_t*)malloc(size_c);
        uint8_t* h_C_cpu = (uint8_t*)malloc(size_c);

        for (size_t i = 0; i < size_a; ++i) {
            h_A[i] = (i % 3 == 0) ? 0 : ((i % 3 == 1) ? 1 : 255);
        }
        for (size_t i = 0; i < size_b; ++i) {
            h_B[i] = ((i * 7) % 3 == 0) ? 0 : (((i * 7) % 3 == 1) ? 1 : 255);
        }

        // CPU reference
        memset(h_C_cpu, 0, size_c);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int l = 0; l < k; ++l) {
                    uint8_t a = h_A[i * k + l];
                    uint8_t b = h_B[l * n + j];
                    if (a != 0 && b != 0) {
                        h_C_cpu[i * n + j] ^= gf_exp[gf_log[a] + gf_log[b]];
                    }
                }
            }
        }

        uint8_t *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_a));
        CUDA_CHECK(cudaMalloc(&d_B, size_b));
        CUDA_CHECK(cudaMalloc(&d_C, size_c));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

        GFGemmHandle handle;
        GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));
        GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));

        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

        int errors = 0;
        for (size_t i = 0; i < size_c; ++i) {
            if (h_C_gpu[i] != h_C_cpu[i]) errors++;
        }
        printf("  Mixed (0,1,255): %d errors\n", errors);
        total_errors += errors;

        gf_gemm_destroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    }

    printf("  Total errors: %d\n", total_errors);
    return total_errors > 0 ? 1 : 0;
}

/**
 * @brief Test different leading dimensions (stride)
 */
int test_leading_dimension() {
    printf("\n=== Test: Leading Dimension (Stride) ===\n");

    int m = 8, n = 8, k = 8;
    int lda = k * 2;  // Padding
    int ldb = n * 2;  // Padding
    int ldc = n * 2;  // Padding

    size_t size_a = m * lda;
    size_t size_b = k * ldb;
    size_t size_c = m * ldc;

    uint8_t* h_A = (uint8_t*)malloc(size_a);
    uint8_t* h_B = (uint8_t*)malloc(size_b);
    uint8_t* h_C_gpu = (uint8_t*)malloc(size_c);
    uint8_t* h_C_cpu = (uint8_t*)malloc(size_c);

    memset(h_A, 0, size_a);
    memset(h_B, 0, size_b);
    memset(h_C_gpu, 0, size_c);
    memset(h_C_cpu, 0, size_c);

    // Fill actual data in padded arrays
    srand(5678);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            h_A[i * lda + j] = rand() % 256;
        }
    }
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            h_B[i * ldb + j] = rand() % 256;
        }
    }

    // CPU reference with same leading dimensions
    uint8_t gf_exp[768], gf_log[256];
    const uint8_t prim_poly = 0x1D;
    uint8_t exp_val = 1;
    for (int log = 0; log < 255; ++log) {
        gf_log[exp_val] = static_cast<uint8_t>(log);
        gf_exp[log] = exp_val;
        gf_exp[log + 255] = exp_val;
        gf_exp[log + 510] = exp_val;
        exp_val = (exp_val << 1) ^ ((exp_val & 0x80) ? prim_poly : 0);
    }
    gf_log[0] = 0;
    gf_exp[0] = 1;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int l = 0; l < k; ++l) {
                uint8_t a = h_A[i * lda + l];
                uint8_t b = h_B[l * ldb + j];
                if (a != 0 && b != 0) {
                    h_C_cpu[i * ldc + j] ^= gf_exp[gf_log[a] + gf_log[b]];
                }
            }
        }
    }

    uint8_t *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_a));
    CUDA_CHECK(cudaMalloc(&d_B, size_b));
    CUDA_CHECK(cudaMalloc(&d_C, size_c));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    GFGemmHandle handle;
    GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));
    GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, lda, d_B, ldb, d_C, ldc, 0));

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (h_C_gpu[i * ldc + j] != h_C_cpu[i * ldc + j]) errors++;
        }
    }
    printf("  Padded leading dimensions: %d errors\n", errors);

    gf_gemm_destroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);

    return errors;
}

/**
 * @brief Test error handling
 */
int test_error_handling() {
    printf("\n=== Test: Error Handling ===\n");

    GFGemmHandle handle = nullptr;
    GFGemmConfig config;
    gf_gemm_config_init_default(&config);
    GF_GEMM_CHECK(gf_gemm_create(&handle, &config));

    int errors = 0;

    // Test 1: NULL handle for mm
    GFGemmError err = gf_gemm_mm(nullptr, 16, 16, 16, nullptr, 16, nullptr, 16, nullptr, 16, 0);
    if (err != GF_GEMM_ERROR_NOT_INITIALIZED) {
        printf("  NULL handle: expected NOT_INITIALIZED, got %s\n", gf_gemm_get_error_string(err));
        errors++;
    } else {
        printf("  NULL handle: OK\n");
    }

    // Test 2: NULL matrix pointers
    err = gf_gemm_mm(handle, 16, 16, 16, nullptr, 16, nullptr, 16, nullptr, 16, 0);
    if (err != GF_GEMM_ERROR_INVALID_VALUE) {
        printf("  NULL matrices: expected INVALID_VALUE, got %s\n", gf_gemm_get_error_string(err));
        errors++;
    } else {
        printf("  NULL matrices: OK\n");
    }

    // Test 3: Invalid dimensions (zero)
    uint8_t dummy;
    err = gf_gemm_mm(handle, 0, 16, 16, &dummy, 16, &dummy, 16, &dummy, 16, 0);
    if (err != GF_GEMM_ERROR_INVALID_VALUE) {
        printf("  Zero m dimension: expected INVALID_VALUE, got %s\n", gf_gemm_get_error_string(err));
        errors++;
    } else {
        printf("  Zero m dimension: OK\n");
    }

    err = gf_gemm_mm(handle, 16, 0, 16, &dummy, 16, &dummy, 16, &dummy, 16, 0);
    if (err != GF_GEMM_ERROR_INVALID_VALUE) {
        printf("  Zero n dimension: expected INVALID_VALUE, got %s\n", gf_gemm_get_error_string(err));
        errors++;
    } else {
        printf("  Zero n dimension: OK\n");
    }

    err = gf_gemm_mm(handle, 16, 16, 0, &dummy, 16, &dummy, 16, &dummy, 16, 0);
    if (err != GF_GEMM_ERROR_INVALID_VALUE) {
        printf("  Zero k dimension: expected INVALID_VALUE, got %s\n", gf_gemm_get_error_string(err));
        errors++;
    } else {
        printf("  Zero k dimension: OK\n");
    }

    // Test 4: NULL handle for compute
    err = gf_gemm_compute(nullptr, 16, 16, 16, nullptr, nullptr, 16, nullptr, 16, nullptr, nullptr, 16, 0);
    if (err != GF_GEMM_ERROR_NOT_INITIALIZED) {
        printf("  NULL handle (compute): expected NOT_INITIALIZED, got %s\n", gf_gemm_get_error_string(err));
        errors++;
    } else {
        printf("  NULL handle (compute): OK\n");
    }

    gf_gemm_destroy(handle);

    printf("  Error handling tests: %d failures\n", errors);
    return errors;
}

int main(int argc, char** argv) {
    printf("=== CUTLASS GF(2^8) GEMM Test Suite ===\n");

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    int failed = 0;
    failed += test_basic_gemm();
    failed += test_identity();
    failed += test_zero_matrix();
    failed += test_various_sizes();
    failed += test_performance();

    // New corner case tests
    failed += test_alpha_beta_scaling();
    failed += test_minimum_size();
    failed += test_non_aligned_sizes();
    failed += test_gf_boundary_values();
    failed += test_leading_dimension();
    failed += test_error_handling();

    printf("\n=== Test Summary ===\n");
    if (failed == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED!\n", failed);
    }

    return failed;
}
