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

    printf("\n=== Test Summary ===\n");
    if (failed == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED!\n", failed);
    }

    return failed;
}
