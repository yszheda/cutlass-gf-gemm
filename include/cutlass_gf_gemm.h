/**
 * @file cutlass_gf_gemm.h
 * @brief CUTLASS-based Galois Field Matrix Multiplication
 *
 * This header provides a high-level interface for performing matrix
 * multiplication over GF(2^8) using the CUTLASS library.
 */

#pragma once

#include <stdint.h>
#include <vector>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
//                     Error Handling
// ============================================================================

/// GFGemm error codes
typedef enum {
    GF_GEMM_SUCCESS = 0,
    GF_GEMM_ERROR_INVALID_VALUE = 1,
    GF_GEMM_ERROR_NOT_INITIALIZED = 2,
    GF_GEMM_ERROR_OUT_OF_MEMORY = 3,
    GF_GEMM_ERROR_CUBLAS_INIT_FAILED = 4,
    GF_GEMM_ERROR_CUDA_KERNEL_FAILED = 5,
    GF_GEMM_ERROR_UNSUPPORTED = 6
} GFGemmError;

/// Get error string
const char* gf_gemm_get_error_string(GFGemmError error);

// ============================================================================
//                     Opaque Handle
// ============================================================================

/// Opaque handle to GFGemm instance
typedef struct GFGemmImpl* GFGemmHandle;

// ============================================================================
//                     Configuration
// ============================================================================

/// Matrix layout type
typedef enum {
    GF_GEMM_ROW_MAJOR = 0,
    GF_GEMM_COL_MAJOR = 1
} GFGemmLayout;

/// Configuration parameters for GFGemm
typedef struct {
    /// Matrix A layout
    GFGemmLayout layout_a;

    /// Matrix B layout
    GFGemmLayout layout_b;

    /// Matrix C layout
    GFGemmLayout layout_c;

    /// Number of CUDA streams for concurrent execution
    int num_streams;

    /// Enable profiling
    int enable_profiling;

    /// Reserved for future use
    int reserved[8];
} GFGemmConfig;

/// Initialize config with default values
void gf_gemm_config_init_default(GFGemmConfig* config);

// ============================================================================
//                     API Functions
// ============================================================================

/**
 * @brief Create a new GFGemm instance
 *
 * @param handle Output handle
 * @param config Configuration parameters (can be NULL for defaults)
 * @return GFGemmError
 */
GFGemmError gf_gemm_create(GFGemmHandle* handle, const GFGemmConfig* config);

/**
 * @brief Destroy a GFGemm instance
 *
 * @param handle Handle to destroy
 * @return GFGemmError
 */
GFGemmError gf_gemm_destroy(GFGemmHandle handle);

/**
 * @brief Perform GF(2^8) matrix multiplication: D = alpha * A * B + beta * C
 *
 * For Galois Field arithmetic:
 * - alpha and beta are GF(2^8) elements (0 or 1 effectively, since other values
 *   would require full GF multiplication)
 * - Addition is XOR
 * - Multiplication uses log/exp tables
 *
 * @param handle GFGemm instance
 * @param m Number of rows of A and C
 * @param n Number of columns of B and C
 * @param k Number of columns of A and rows of B
 * @param alpha Scalar multiplier for A*B (can be NULL for 1)
 * @param A Pointer to matrix A (size: m*k)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (size: k*n)
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C (can be NULL for 0)
 * @param C Pointer to matrix C (input/output, size: m*n)
 * @param ldc Leading dimension of C
 * @param stream CUDA stream (can be 0 for default)
 * @return GFGemmError
 */
GFGemmError gf_gemm_compute(GFGemmHandle handle,
                            int m, int n, int k,
                            const uint8_t* alpha,
                            const uint8_t* A, int lda,
                            const uint8_t* B, int ldb,
                            const uint8_t* beta,
                            uint8_t* C, int ldc,
                            cudaStream_t stream);

/**
 * @brief Perform GF(2^8) matrix multiplication: D = A * B
 *
 * Simplified version without alpha/beta scalars.
 *
 * @param handle GFGemm instance
 * @param m Number of rows of A and C
 * @param n Number of columns of B and C
 * @param k Number of columns of A and rows of B
 * @param A Pointer to matrix A (size: m*k)
 * @param B Pointer to matrix B (size: k*n)
 * @param C Pointer to matrix C (output, size: m*n)
 * @param stream CUDA stream (can be 0 for default)
 * @return GFGemmError
 */
GFGemmError gf_gemm_mm(GFGemmHandle handle,
                       int m, int n, int k,
                       const uint8_t* A, int lda,
                       const uint8_t* B, int ldb,
                       uint8_t* C, int ldc,
                       cudaStream_t stream);

/**
 * @brief Synchronize GFGemm instance
 *
 * Waits for all pending operations to complete.
 *
 * @param handle GFGemm instance
 * @return GFGemmError
 */
GFGemmError gf_gemm_synchronize(GFGemmHandle handle);

// ============================================================================
//                     Utility Functions
// ============================================================================

/**
 * @brief Initialize GF(2^8) lookup tables on device
 *
 * @param gf_exp Device pointer to exp table (size >= 768 bytes)
 * @param gf_log Device pointer to log table (size >= 256 bytes)
 * @return cudaError_t
 */
cudaError_t gf_init_tables_device(uint8_t* gf_exp, uint8_t* gf_log);

/**
 * @brief Generate random GF(2^8) matrix
 *
 * @param matrix Output matrix
 * @param size Number of elements
 * @param seed Random seed
 */
void gf_random_matrix(uint8_t* matrix, size_t size, unsigned int seed);

/**
 * @brief Print GF(2^8) matrix
 *
 * @param matrix Matrix to print
 * @param rows Number of rows
 * @param cols Number of columns
 * @param name Matrix name
 */
void gf_print_matrix(const uint8_t* matrix, int rows, int cols, const char* name);

/**
 * @brief Verify matrix multiplication result
 *
 * Computes reference result on CPU and compares with GPU result.
 *
 * @param A Matrix A
 * @param B Matrix B
 * @param C GPU result
 * @param m Number of rows
 * @param n Number of columns
 * @param k Number of reduction elements
 * @return 0 if results match, -1 otherwise
 */
int gf_verify_gemm(const uint8_t* A, const uint8_t* B, const uint8_t* C,
                   int m, int n, int k);

#ifdef __cplusplus
} // extern "C"
#endif

// ============================================================================
//                     C++ Wrapper Class
// ============================================================================

#ifdef __cplusplus

#include <memory>
#include <stdexcept>

namespace gf {

/// C++ wrapper for GFGemm
class Gemm {
public:
    /// Default constructor
    Gemm() {
        GFGemmConfig config;
        gf_gemm_config_init_default(&config);
        init(&config);
    }

    /// Constructor with config
    explicit Gemm(const GFGemmConfig& config) {
        init(&config);
    }

    /// Destructor
    ~Gemm() {
        if (handle_) {
            gf_gemm_destroy(handle_);
        }
    }

    /// Move constructor
    Gemm(Gemm&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    /// Move assignment
    Gemm& operator=(Gemm&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                gf_gemm_destroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    // Disable copy
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    /// Perform matrix multiplication: C = A * B
    void mm(int m, int n, int k,
            const uint8_t* A, int lda,
            const uint8_t* B, int ldb,
            uint8_t* C, int ldc,
            cudaStream_t stream = 0) {
        auto err = gf_gemm_mm(handle_, m, n, k, A, lda, B, ldb, C, ldc, stream);
        if (err != GF_GEMM_SUCCESS) {
            throw std::runtime_error(gf_gemm_get_error_string(err));
        }
    }

    /// Perform matrix multiplication with alpha/beta: D = alpha*A*B + beta*C
    void compute(int m, int n, int k,
                 const uint8_t* alpha,
                 const uint8_t* A, int lda,
                 const uint8_t* B, int ldb,
                 const uint8_t* beta,
                 uint8_t* C, int ldc,
                 cudaStream_t stream = 0) {
        auto err = gf_gemm_compute(handle_, m, n, k, alpha, A, lda, B, ldb,
                                   beta, C, ldc, stream);
        if (err != GF_GEMM_SUCCESS) {
            throw std::runtime_error(gf_gemm_get_error_string(err));
        }
    }

    /// Synchronize
    void synchronize() {
        auto err = gf_gemm_synchronize(handle_);
        if (err != GF_GEMM_SUCCESS) {
            throw std::runtime_error(gf_gemm_get_error_string(err));
        }
    }

    /// Get native handle
    GFGemmHandle native_handle() const { return handle_; }

private:
    void init(const GFGemmConfig* config) {
        auto err = gf_gemm_create(&handle_, config);
        if (err != GF_GEMM_SUCCESS) {
            throw std::runtime_error(gf_gemm_get_error_string(err));
        }
    }

    GFGemmHandle handle_ = nullptr;
};

} // namespace gf

#endif // __cplusplus
