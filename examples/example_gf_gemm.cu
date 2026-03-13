/**
 * @file example_gf_gemm.cu
 * @brief Example usage of CUTLASS GF Gemm library
 *
 * This example demonstrates how to perform matrix multiplication
 * over GF(2^8) using the GFGemm library.
 */

#include "cutlass_gf_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define GF_GEMM_CHECK(call) \
    do { \
        GFGemmError err = call; \
        if (err != GF_GEMM_SUCCESS) { \
            fprintf(stderr, "GF Gemm error at %s:%d: %s\n", \
                    __FILE__, __LINE__, gf_gemm_get_error_string(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief Print a small matrix
 */
void print_matrix(const uint8_t* matrix, int rows, int cols, const char* name) {
    printf("Matrix %s (%dx%d):\n", name, rows, cols);
    int max_print_rows = 8;
    int max_print_cols = 8;

    for (int i = 0; i < rows && i < max_print_rows; ++i) {
        printf("  ");
        for (int j = 0; j < cols && j < max_print_cols; ++j) {
            printf("%3d ", matrix[i * cols + j]);
        }
        if (cols > max_print_cols) printf("...");
        printf("\n");
    }
    if (rows > max_print_rows) {
        printf("  ...\n");
    }
}

/**
 * @brief CPU reference implementation for verification
 */
void gf_gemm_cpu(const uint8_t* A, const uint8_t* B, uint8_t* C,
                 int m, int n, int k) {
    // Build GF tables
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

    // Compute C = A * B over GF(2^8)
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

int main(int argc, char** argv) {
    printf("=== CUTLASS GF(2^8) Matrix Multiplication Example ===\n\n");

    // Matrix dimensions
    int m = 64;   // rows of A and C
    int n = 128;  // columns of B and C
    int k = 96;   // columns of A and rows of B

    // Parse command line arguments
    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) k = atoi(argv[3]);

    printf("Matrix dimensions:\n");
    printf("  A: %d x %d\n", m, k);
    printf("  B: %d x %d\n", k, n);
    printf("  C: %d x %d\n\n", m, n);

    // Allocate host memory
    size_t size_a = m * k * sizeof(uint8_t);
    size_t size_b = k * n * sizeof(uint8_t);
    size_t size_c = m * n * sizeof(uint8_t);

    uint8_t* h_A = (uint8_t*)malloc(size_a);
    uint8_t* h_B = (uint8_t*)malloc(size_b);
    uint8_t* h_C_gpu = (uint8_t*)malloc(size_c);
    uint8_t* h_C_cpu = (uint8_t*)malloc(size_c);

    // Initialize matrices with random values
    printf("Initializing matrices with random values...\n");
    srand(42);
    for (size_t i = 0; i < m * k; ++i) h_A[i] = rand() % 256;
    for (size_t i = 0; i < k * n; ++i) h_B[i] = rand() % 256;

    // Print small sample of input matrices
    if (m <= 8 && k <= 8) {
        print_matrix(h_A, m, k, "A");
    } else {
        print_matrix(h_A, 8, 8, "A (sample)");
    }

    if (k <= 8 && n <= 8) {
        print_matrix(h_B, k, n, "B");
    } else {
        print_matrix(h_B, 8, 8, "B (sample)");
    }

    // Allocate device memory
    uint8_t *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_a));
    CUDA_CHECK(cudaMalloc(&d_B, size_b));
    CUDA_CHECK(cudaMalloc(&d_C, size_c));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    // Create GFGemm instance
    printf("\nInitializing GFGemm...\n");
    GFGemmHandle handle;
    GF_GEMM_CHECK(gf_gemm_create(&handle, nullptr));

    // Create CUDA event for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up run
    printf("Warm-up run...\n");
    GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark run
    printf("Running benchmark...\n");
    cudaEventRecord(start, 0);
    int num_iterations = 10;
    for (int i = 0; i < num_iterations; ++i) {
        GF_GEMM_CHECK(gf_gemm_mm(handle, m, n, k, d_A, k, d_B, n, d_C, n, 0));
    }
    cudaEventRecord(stop, 0);
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_ms, start, stop));
    float avg_time_ms = elapsed_time_ms / num_iterations;

    printf("Average execution time: %.3f ms\n", avg_time_ms);

    // Calculate GFLOPS (for GF operations, we count GMACS)
    float gflops = (2.0f * m * n * k) / (avg_time_ms * 1e6f);
    printf("Performance: %.2f GOPS (Giga Operations per Second)\n", gflops);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

    // Compute CPU reference
    printf("\nComputing CPU reference...\n");
    gf_gemm_cpu(h_A, h_B, h_C_cpu, m, n, k);

    // Verify results
    printf("Verifying results...\n");
    int errors = 0;
    for (size_t i = 0; i < m * n; ++i) {
        if (h_C_gpu[i] != h_C_cpu[i]) {
            errors++;
            if (errors <= 10) {
                printf("Mismatch at [%zu]: GPU=%d, CPU=%d\n",
                       i, h_C_gpu[i], h_C_cpu[i]);
            }
        }
    }

    if (errors == 0) {
        printf("✓ Results verified successfully!\n");
    } else {
        printf("✗ Verification failed with %d errors out of %zu elements\n",
               errors, m * n);
    }

    // Print sample of result
    print_matrix(h_C_gpu, m, n, "C = A * B");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    gf_gemm_destroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    printf("\n=== Example completed ===\n");

    return errors == 0 ? 0 : 1;
}
