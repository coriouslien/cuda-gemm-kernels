<pre>
A structured through the dense CuTe material: small chunks, in program order, one at a time. 
I move to the next chunk only when the current one is clear. All source code line numbers are 
shown in the file I attached. 
</pre>
|Chunk	|Lines	|Topic|
|:------|:------|:----|
|0	|whole file|	The big picture: what the program does, end to end (below)|
|1	|L467–581, L446–464	|main and gemm: host data, which kernel runs|
|2	|L352–380	|gemm_tn: problem shape, strides, CTA tile|
|3	|L382–384	|shared-memory layouts: tile_to_shape, the GMMA atom, swizzle|
|4|	L386–392	|the copy: make_tiled_copy, the TV layout|
|5	|L394, L412–443|	the MMA atom and the launch|
|6	|L18–26, L28–81	|SharedStorage, the kernel signature, global and shared tensors|
|7	|L83–100	|copy partitioning|
|8	L123–143	|MMA partitioning, descriptors, accumulators|
|9	|L178–251	|the main loop|
|10	|L253–258	|the epilogue|
|11	|—	|verify: print_layouts.cu, then the SASS|
<pre>
Chunk 0: The big picture
What the program computes

A matrix multiplication in FP16:

C = A · Bᵀ          A is M × K,  B is N × K,  C is M × N

In CuTe’s convention, both A and B are stored with K as their second mode, which is why the product is written with Bᵀ. 
Default sizes (L487–497): M = N = 5120, K = 4096.

How the work is split

C is cut into 128 × 128 tiles. Each thread block (CTA) computes one tile:

grid = (5120/128) × (5120/128) = 40 × 40 = 1600 CTAs
each CTA: 128 threads = 1 warpgroup

To compute its tile, a CTA walks along K in chunks of 64: 4096 / 64 = 64 loop iterations.

The path the data takes inside one CTA
global memory           shared memory               registers            global memory
A tile (128×64) ─┐
                 ├─cp.async─►  sA, sB              ─wgmma─►  tCrC         ─axpby─►  C tile
B tile (128×64) ─┘            (3 stages, swizzled)   (reads smem via   (128×128 accumulator,     (128×128)
                                                      descriptors)       spread over 128 threads)
        └──────────── repeated 64 times along K ────────────┘
The four parts of the file, in calling order
main (L467)                     host: make A, B, C; call gemm once; then time 100 calls
 └─ gemm (L449)                 host: pick NT or TN from transA/transB
     ├─ gemm_nt (L264)          host: build shapes, layouts, copy, MMA; launch kernel
     └─ gemm_tn (L356)          host: same, TN variant
         └─ gemm_device (L36)   device: the kernel — one CTA computes one 128×128 tile of C

The host functions (gemm_nt, gemm_tn) do nothing but describe: sizes, layouts, the copy pattern and the MMA instruction, all as CuTe types. 
The kernel uses those descriptions. Since almost everything is a compile-time type, the description is what appears in the long demangled 
kernel names in your build log.

I'm continuing my study of CUTLASS's Hopper tutorial wgmma_sm90.cu (second pass, chunk by chunk).

Setup
- Compiling (not running) for sm_90a on my RTX 5080/5050 (SM120), CUDA 12.9, Release build
  (-O3 -DNDEBUG via CMAKE_BUILD_TYPE). No C7510 warning in Release.
- ptxas: TN kernel 112 regs, NT kernel 114 regs, 1 barrier, 0 spills; also cub::EmptyKernel (from Thrust).

Confirmed facts so far
- Loads use cp.async (SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, LDGSTS), NOT TMA.
- CTA tile 128x128x64, 3 pipeline stages, 128 threads (1 warpgroup), 98,304 B dynamic smem.
- TN: Layout_K_SW128_Atom -> ((8,16),(64,1),(1,3)):((64,512),(1,0),(0,8192)), Swizzle<3,4,3>.
- Atom in bits (8,1024):(1024,1) -> upcast to half_t (8,64):(64,1).
- TN copy TV layout ((8,16),8):((128,1),16), tile (16,64): thread t copies row t/8, K = 8*(t%8)..+7.
- MMA: MMA_64x64x16_F16F16F16_SS, 16 wgmma per k-tile; tCrA/tCrB hold smem descriptors.
- Default run uses the NT kernel (transA='N', transB='T').
- Main loop: cp_async_wait<0>() and warpgroup_wait<0>() -> no overlap of load and compute.

Where I am
- Plan: 1 main/gemm, 2 gemm_tn shapes/strides, 3 smem layouts, 4 copy, 5 MMA+launch,
  6 SharedStorage/kernel tensors, 7 copy partition, 8 MMA partition, 9 main loop,
  10 epilogue, 11 print_layouts.cu + SASS.


</pre>
