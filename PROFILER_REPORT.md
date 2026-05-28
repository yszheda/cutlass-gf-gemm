# GF(2^8) GEMM Profiler Report

**Date:** 2026-05-28
**Branch:** master

## Summary

| Server | GPU | Compute Cap | Best Custom (GMACS) | Best CUTLASS (GMACS) | 1024³ Time (ms) Custom | 1024³ Time (ms) CUTLASS |
|--------|-----|-------------|---------------------|----------------------|------------------------|------------------------|
| Thor   | NVIDIA Thor | 11.0 | 0.36 | 0.36 | 5983.370 | 5983.258 |
| A40    | NVIDIA A40  | 8.6  | 95.81 | 96.62 | 22.413 | 22.225 |

**Key findings:**
- Both backends produce **identical numerical results** on both servers (0 mismatches at 64³)
- Custom and CUTLASS backends perform **virtually identically** on each server (within 1%)
- A40 is **~270x faster** than Thor for 1024³ matrix multiplication
- Thor shows minimal GMACS scaling with size (0.26 → 0.36), suggesting a driver/compilation bottleneck
- A40 scales well from small to large matrices (15.9 → 95.8 GMACS)

## Profiler Methodology

- **Tool:** CUDA Events (`cudaEventRecord`, `cudaEventElapsedTime`)
- **Matrix sizes:** 64³, 128³, 256³, 512³, 1024³ (square, M=N=K)
- **Iterations:** 10 per size (64-256), 5 per size (512-1024)
- **Warm-up:** 1 run before timing
- **GMACS:** 2 × M × N × K / avg_time_ms × 1e-6
- **Bandwidth:** (sizeof(A) + sizeof(B) + sizeof(C)) × iters / elapsed_time
- **Nsight Compute:** Not available on either server

## Raw Metrics

### Thor (10.190.0.91) — NVIDIA Thor, CC 11.0

```
--- Custom Backend ---
      Size    Time(ms)       GMACS      BW(GB/s)
  --------  ----------  ----------  ------------
    64x  64       1.994        0.26          0.01
   128x 128      14.793        0.28          0.00
   256x 256      95.216        0.35          0.00
   512x 512     759.385        0.35          0.00
  1024x1024    5983.370        0.36          0.00

--- CUTLASS Backend ---
      Size    Time(ms)       GMACS      BW(GB/s)
  --------  ----------  ----------  ------------
    64x  64       2.070        0.25          0.01
   128x 128      14.883        0.28          0.00
   256x 256      95.294        0.35          0.00
   512x 512     759.510        0.35          0.00
  1024x1024    5983.258        0.36          0.00
```

**Observations:**
- 1024³ takes ~6 seconds — orders of magnitude slower than expected
- GMACS barely increases with matrix size (0.26 → 0.36) — not scaling
- Custom and CUTLASS backends are identical (same kernel path used)
- **Suspect:** CUDA 11.0 (CC 11.0) is a new architecture. The kernel may not have an optimized PTX path for this architecture. The binary was compiled with `-DCMAKE_CUDA_ARCHITECTURES=70 75 80 86 90` which does not include `110`. This causes JIT compilation from PTX at runtime, which is slow.

### A40 (szc-td04) — NVIDIA A40, CC 8.6

```
--- Custom Backend ---
      Size    Time(ms)       GMACS      BW(GB/s)
  --------  ----------  ----------  ------------
    64x  64       0.033       15.90          0.37
   128x 128       0.063       66.17          0.78
   256x 256       0.462       72.61          0.43
   512x 512       2.980       90.08          0.26
  1024x1024      22.413       95.81          0.14

--- CUTLASS Backend ---
      Size    Time(ms)       GMACS      BW(GB/s)
  --------  ----------  ----------  ------------
    64x  64       0.037       14.26          0.33
   128x 128       0.067       62.15          0.73
   256x 256       0.466       71.94          0.42
   512x 512       2.958       90.74          0.27
  1024x1024      22.225       96.62          0.14
```

**Observations:**
- Clean scaling: 64³ → 1024³ shows 6x GMACS improvement (15.9 → 95.8)
- Both backends within 1% of each other at every size
- Bandwidth decreases at larger sizes as kernel becomes compute-bound
- Peak performance: ~96 GMACS at 1024³
- Custom is slightly faster at smaller sizes (64³: 0.033ms vs 0.037ms)
- CUTLASS is slightly faster at larger sizes (1024³: 22.225ms vs 22.413ms)

## Analysis

### Correctness

Both backends produce **bit-identical results**. The 64×64×64 comparison test reports 0 mismatches on both servers. All 13 existing tests pass on both servers.

### Memory vs Compute Bound

On A40, the bandwidth metric decreases from 0.37 GB/s (64³) to 0.14 GB/s (1024³), while GMACS increases from 15.9 to 95.8. This indicates:
- **Small matrices:** Kernel is memory-bound (overhead dominated, low compute utilization)
- **Large matrices (1024³):** Kernel transitions toward compute-bound behavior (higher GMACS, lower relative bandwidth)
- The kernel uses log/exp table lookups, which are memory-bound by nature. The tables fit in constant memory, so the bottleneck is the lookup latency rather than arithmetic throughput.

### Scaling Behavior

**A40 scaling:**
- 64³ → 128³: 4.2x GMACS (15.9 → 66.2) — excellent, kernel fills more of the GPU
- 128³ → 256³: 1.1x — diminishing returns as occupancy saturates
- 256³ → 512³: 1.2x — gradual improvement from better cache utilization
- 512³ → 1024³: 1.06x — approaching asymptotic performance

**Thor scaling:**
- Nearly flat across all sizes — strongly suggests JIT compilation fallback
- The kernel binary was not compiled for CC 11.0, so it's being JIT-compiled from generic PTX

### Bottleneck Identification

1. **Primary bottleneck (both servers):** Log/exp table lookup latency. Each GF multiply requires two table lookups and one conditional branch, which creates instruction divergence in SIMD warps.
2. **Secondary bottleneck (small matrices):** Kernel launch overhead and shared memory tile loading for matrices smaller than the 16×16 tile size.
3. **Thor-specific:** Missing native code generation for CC 11.0 architecture. Recompile with `-DCMAKE_CUDA_ARCHITECTURES=110`.

## Optimization Recommendations

1. **Add CC 11.0 to CMake defaults** — Thor needs native code generation. Current default is `70 75 80 86 90`, missing `110`. Add `-DCMAKE_CUDA_ARCHITECTURES=110` on Thor.

2. **Use bitwise GF multiplication** — For small matrices, the log/exp table approach has high latency. A bitwise multiplication (8 XOR + 8 AND operations) avoids table lookups and may be faster for sizes ≤ 256.

3. **Increase tile size to 32×32** — Current 16×16 tiles limit parallelism. A 32×32 tile with 1024 threads per block would better utilize A40's 84 SMs (1024³ uses only ~4096 threads in 256 blocks, under-utilizing the GPU).

4. **Use vectorized loads (uint4)** — Loading 4 elements at a time with `uint4` would reduce memory transactions by 4× for the tile loading phase, improving bandwidth utilization.

5. **Consider shared memory for GF tables** — Currently tables are in constant memory. For larger threadblocks, caching tables in shared memory (one copy per block) could reduce lookup latency compared to constant memory broadcasts.

6. **Profile with Nsight Compute** — Neither server has `ncu` available. Running `ncu --set full` would identify specific bottleneck metrics: active warps/SM, L1 hit rate, instruction issue rate.

## Backend Comparison Conclusion

Both backends are functionally identical and produce matching results. The CUTLASS backend is a viable drop-in replacement for the custom kernel. The decision between them should be based on:
- **CUTLASS backend:** Better for future integration with real CUTLASS templates, follows CUTLASS patterns
- **Custom backend:** Simpler, slightly faster at small sizes on A40

For production use on A40 at 1024³, CUTLASS is marginally faster (96.62 vs 95.81 GMACS). At smaller sizes, custom is slightly faster. The difference is within measurement noise.
