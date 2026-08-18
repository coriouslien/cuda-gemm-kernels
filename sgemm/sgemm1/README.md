<pre>
# sgemm1 — Pipelined HGEMM with CuTe (SM120, RTX 5080)

CuTe-based pipelined half-precision GEMM (TN layout) using 
SM80_16x8x16 MMA atoms with software-pipelined cp.async 
and LDSM. Adapted from the NVIDIA CUTLASS CuTe tutorial 
and validated on RTX 5080 (SM120, CC 12.0).

## Hardware & Environment
| Item      | Value                                   |
|-----------|-----------------------------------------|
| GPU       | NVIDIA GeForce RTX 5080 (SM120, 84 SMs) |
| CUDA      | 12.8                                    |
| OS        | Ubuntu 24.04                            |
| Profiler  | Nsight Compute (ncu)                    |

## Build

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/sgemm 5120 5120 4096 T N

## Performance
| Configuration                   | GFlop/s  | Time (ms) |
|---------------------------------|----------|-----------|
| CUTE HGEMM (5120×5120×4096, TN) | 187,086  | 1.148     |

Verified correct against cuBLAS: max element-wise error < 1.0 
(FP16 accumulation, expected rounding).

## Key Implementation Details
- **MMA atom**: SM80_16x8x16_F16F16F16F16_TN (Tensor Core)
- **Tile**: 128×128×64 with 3-stage pipeline
- **Smem layout**: Swizzle<3,3,3> k-major for bank-conflict-free LDSM
- **Gmem→Smem**: SM80_CP_ASYNC_CACHEALWAYS<uint128_t> (128-bit async)
- **Smem→Reg**: SM75_U32x4_LDSM_N
- **Register count**: 168/thread, 0 spills
- **Occupancy**: 8.3% (limited by shared memory: 1 block/SM)

## NCU Profiling Results
See [profiling/](profiling/) for full Nsight Compute screenshots.

| Metric                  | Value                      |
|-------------------------|----------------------------|
| Compute (SM) throughput | 82.19%                     |
| Achieved occupancy      | 8.34%                      |
| Active warps/scheduler  | 1.00 (of 12 max)           |
| No-eligible cycles      | 84.32%                     |
| Dominant stall          | Stall Wait (70.5%)         |
| Occupancy limiter       | Shared memory (1 block/SM) |

**Primary bottleneck**: shared memory usage limits to 1 block/SM, 
leaving only 4 active warps per SM and 84% scheduler idle cycles.
Next step: reduce smem footprint to enable 2+ concurrent blocks.

</pre>
