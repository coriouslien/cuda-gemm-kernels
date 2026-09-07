**The following is sgemm_2.cu compile output**

[ 50%] Building CUDA object CMakeFiles/sgemm_2.dir/sgemm_2.cu.o

ptxas info : 47 bytes gmem

ptxas info : Compiling entry function '_ZN3cub18CUB_200802_SM_120011EmptyKernelIvEEvv' for 'sm_120'

ptxas info : Function properties for _ZN3cub18CUB_200802_SM_120011EmptyKernelIvEEvv

0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads

ptxas info : Used 4 registers, used 0 barriers

ptxas info : Compile time = 0.895 ms

ptxas info : Compiling entry function 'Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi8EEEEEEfNS1_IJiNS3_ILi1EEEEEENS0_6LayoutINS1_IJS4_S5_EEENS1_IJS7_NS3_ILi129EEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_13UniversalCopyIffEEfEEENS9_INS1_IJNS1_IJS5_NS3_ILi32EEEEEES7_EEENS1_IJNS1_IJSJ_S7_EEENS3_ILi0EEEEEEEENS1_IJSJ_S5_EEEEEfS8_SD_SR_fNS1_IJS7_iEEENS9_INS1_IJS4_S4_EEENS1_IJS7_S4_EEEEENS0_8TiledMMAINS0_8MMA_AtomIJNS0_12UniversalFMAIffffEEEEENS9_INS1_IJNS3_ILi16EEES11_S7_EEENS1_IJS7_S11_SN_EEEEENS1_IJNS0_10UnderscoreES15_S15_EEEEEffEvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_T14' for 'sm_120'

ptxas info : Function properties for Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi8EEEEEEfNS1_IJiNS3_ILi1EEEEEENS0_6LayoutINS1_IJS4_S5_EEENS1_IJS7_NS3_ILi129EEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_13UniversalCopyIffEEfEEENS9_INS1_IJNS1_IJS5_NS3_ILi32EEEEEES7_EEENS1_IJNS1_IJSJ_S7_EEENS3_ILi0EEEEEEEENS1_IJSJ_S5_EEEEEfS8_SD_SR_fNS1_IJS7_iEEENS9_INS1_IJS4_S4_EEENS1_IJS7_S4_EEEEENS0_8TiledMMAINS0_8MMA_AtomIJNS0_12UniversalFMAIffffEEEEENS9_INS1_IJNS3_ILi16EEES11_S7_EEENS1_IJS7_S11_SN_EEEEENS1_IJNS0_10UnderscoreES15_S15_EEEEEffEvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_T14

0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads

ptxas info : Used 128 registers, used 1 barriers, 8248 bytes smem

ptxas info : Compile time = 29.680 ms

ptxas info : Compiling entry function 'Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi8EEEEEEfNS1_IJNS3_ILi1EEEiEEENS0_6LayoutINS1_IJS4_S5_EEENS1_IJS7_S4_EEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_13UniversalCopyIN7cutlass9uint128_tESH_EEfEEENS9_INS1_IJNS3_ILi256EEENS3_ILi4EEEEEENS1_IJSL_S7_EEEEESA_EEfS8_SC_SP_fS8_NS9_INS1_IJS4_S4_EEESB_EENS0_8TiledMMAINS0_8MMA_AtomIJNS0_12UniversalFMAIffffEEEEENS9_INS1_IJNS3_ILi16EEESX_S7_EEENS1_IJS7_SX_NS3_ILi0EEEEEEEENS1_IJNS0_10UnderscoreES12_S12_EEEEEffEvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_T14' for 'sm_120'

ptxas info : Function properties for Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi8EEEEEEfNS1_IJNS3_ILi1EEEiEEENS0_6LayoutINS1_IJS4_S5_EEENS1_IJS7_S4_EEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_13UniversalCopyIN7cutlass9uint128_tESH_EEfEEENS9_INS1_IJNS3_ILi256EEENS3_ILi4EEEEEENS1_IJSL_S7_EEEEESA_EEfS8_SC_SP_fS8_NS9_INS1_IJS4_S4_EEESB_EENS0_8TiledMMAINS0_8MMA_AtomIJNS0_12UniversalFMAIffffEEEEENS9_INS1_IJNS3_ILi16EEESX_S7_EEENS1_IJS7_SX_NS3_ILi0EEEEEEEENS1_IJNS0_10UnderscoreES12_S12_EEEEEffEvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_T14

0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads

ptxas info : Used 106 registers, used 1 barriers, 8192 bytes smem

ptxas info : Compile time = 82.525 ms

[100%] Linking CUDA executable sgemm_2

[100%] Built target sgemm_2 

**This is execution out put shown below:**

build/sgemm_2

M = 5120

N = 5120

K = 4096

C = A^N B^T

Using device 0: NVIDIA GeForce RTX 5080 (SM120, 84 SMs)

CUTE_GEMM: [25129.7]GFlop/s (8.5456)ms 


1. The Operation and Layout
$C = A^N B^T$: This indicates it is computing $C = A B^T$, where matrix $A$ is Non-transposed and matrix $B$ is Transposed. Why this is good: This specific layout is highly cache-friendly. It allows threads to read contiguous blocks of memory along the inner $K$-dimension for both matrices, which maximizes memory bandwidth utilization and perfectly aligns with the vectorized uint128_t memory loads we saw in the compiler output.
2. Performance and Hardware Utilization The Math: A standard matrix multiplication requires $2 \times M \times N \times K$ floating-point operations. For the dimensions ($5120 \times 5120 \times 4096$), that is roughly 214.7 billion FLOPs. Completing this in 8.54 ms yields the 25.1 TFLOP/s throughput reported by CuTe.The Baseline: The RTX 5080 (SM120, 84 SMs) has a theoretical peak FP32 throughput of roughly 56.3 TFLOP/s (depending slightly on the specific AIB card's boost clocks).  The Verdict: The kernel is achieving roughly 44.6% of the GPU's theoretical peak FP32 performance.3. Where it Stands Hitting ~45% of theoretical peak compute on the own custom CuTe kernel using standard FMA math (non-Tensor Core) is a very solid starting point. It confirms the memory layouts are sensible and it isn't severely bottlenecked by latency. However, heavily hand-tuned libraries like cuBLAS will typically push closer to 70–85% of peak for standard FP32 math by relying on deep software pipelining and highly aggressive warp-level scheduling.
