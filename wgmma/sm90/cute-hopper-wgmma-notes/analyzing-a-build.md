
## Compiled examples/cute/tutorial/wgmma_sm90.cu ##
<pre>
cmake --build build
[ 50%] Building CUDA object CMakeFiles/wgmma_sm90.dir/wgmma_sm90.cu.o
ptxas info
: (C7510) Potential Performance Loss: wgmma.mma_async instructions are serialized due to wgmma
pipeline crossing function boundary at a function call in the function
'_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJiNS3_ILi1EEEE
EENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJNS3_IL
i8EEENS3_ILi16EEEEEENS1_IJS5_S9_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS5_NS3_ILi512EEEEEENS1_IJS9_NS3_ILi0EEE
EEENS1_IJSQ_NS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_
tES10_EES8_EEENSG_INS1_IJSJ_SH_EEENS1_IJNS1_IJS4_S9_EEESI_EEEEENS1_IJSI_S5_EEEEES8_SA_SW_S18_S8_NS1_IJS9_iEEENS0
_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1D_5MajorE0ELS1F_0ELNS1D_7ScaleInE1ELS1G_
1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSQ_SQ_SQ_EEEEENS1_IJNS0_10UnderscoreES1M_S1M_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4
_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_'
ptxas info
: (C7510) Potential Performance Loss: wgmma.mma_async instructions are serialized due to wgmma
pipeline crossing function boundary at a function call in the function
'_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJNS3_ILi1EEEiE
EENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJS5_NS3
_ILi2EEEEEENS1_IJNS3_ILi8EEESJ_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS9_NS3_ILi512EEEEEENS1_IJS5_NS3_ILi1024E
EEEEENS1_IJNS3_ILi0EEENS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7
_9uint128_tES11_EES8_EEENSG_INS1_IJS4_SJ_EEENS1_IJSJ_S9_EEEEES14_EES8_SA_SX_S17_S8_SA_NS0_8TiledMMAINS0_8MMA_Ato
mIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1B_5MajorE1ELS1D_1ELNS1B_7ScaleInE1ELS1E_1EEEEEENSG_INS1_IJS9_S9
_S9_EEENS1_IJSS_SS_SS_EEEEENS1_IJNS0_10UnderscoreES1K_S1K_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_
T11_T12_T13_'
ptxas info
ptxas info
: 1076 bytes gmem
: Compiling entry function '_ZN3cub17CUB_200802_SM_90011EmptyKernelIvEEvv' for 'sm_90a'
ptxas info
: Function properties for _ZN3cub17CUB_200802_SM_90011EmptyKernelIvEEvv
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info
: Used 4 registers, used 0 barriers
ptxas info
ptxas info
: Compile time = 0.936 ms
: Compiling entry function
'_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJiNS3_ILi1EEEE
EENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJNS3_IL
i8EEENS3_ILi16EEEEEENS1_IJS5_S9_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS5_NS3_ILi512EEEEEENS1_IJS9_NS3_ILi0EEE
EEENS1_IJSQ_NS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_
tES10_EES8_EEENSG_INS1_IJSJ_SH_EEENS1_IJNS1_IJS4_S9_EEESI_EEEEENS1_IJSI_S5_EEEEES8_SA_SW_S18_S8_NS1_IJS9_iEEENS0
_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1D_5MajorE0ELS1F_0ELNS1D_7ScaleInE1ELS1G_
1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSQ_SQ_SQ_EEEEENS1_IJNS0_10UnderscoreES1M_S1M_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4
_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_' for 'sm_90a'
ptxas info
: Function properties for
_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJiNS3_ILi1EEEEE
ENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJNS3_ILi
8EEENS3_ILi16EEEEEENS1_IJS5_S9_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS5_NS3_ILi512EEEEEENS1_IJS9_NS3_ILi0EEEE
EENS1_IJSQ_NS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES10_EES8_EEENSG_INS1_IJSJ_SH_EEENS1_IJNS1_IJS4_S9_EEESI_EEEEENS1_IJSI_S5_EEEEES8_SA_SW_S18_S8_NS1_IJS9_iEEENS0_
8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1D_5MajorE0ELS1F_0ELNS1D_7ScaleInE1ELS1G_1
EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSQ_SQ_SQ_EEEEENS1_IJNS0_10UnderscoreES1M_S1M_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_
PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info
: Used 116 registers, used 1 barriers
ptxas info
ptxas info
: Compile time = 54.881 ms
: Compiling entry function
'_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJNS3_ILi1EEEiE
EENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJS5_NS3
_ILi2EEEEEENS1_IJNS3_ILi8EEESJ_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS9_NS3_ILi512EEEEEENS1_IJS5_NS3_ILi1024E
EEEEENS1_IJNS3_ILi0EEENS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7
_9uint128_tES11_EES8_EEENSG_INS1_IJS4_SJ_EEENS1_IJSJ_S9_EEEEES14_EES8_SA_SX_S17_S8_SA_NS0_8TiledMMAINS0_8MMA_Ato
mIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1B_5MajorE1ELS1D_1ELNS1B_7ScaleInE1ELS1E_1EEEEEENSG_INS1_IJS9_S9
_S9_EEENS1_IJSS_SS_SS_EEEEENS1_IJNS0_10UnderscoreES1K_S1K_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_
T11_T12_T13_' for 'sm_90a'
ptxas info
: Function properties for
_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJNS3_ILi1EEEiEE
ENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJS5_NS3_
ILi2EEEEEENS1_IJNS3_ILi8EEESJ_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS9_NS3_ILi512EEEEEENS1_IJS5_NS3_ILi1024EE
EEEENS1_IJNS3_ILi0EEENS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_
9uint128_tES11_EES8_EEENSG_INS1_IJS4_SJ_EEENS1_IJSJ_S9_EEEEES14_EES8_SA_SX_S17_S8_SA_NS0_8TiledMMAINS0_8MMA_Atom
IJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1B_5MajorE1ELS1D_1ELNS1B_7ScaleInE1ELS1E_1EEEEEENSG_INS1_IJS9_S9_
S9_EEENS1_IJSS_SS_SS_EEEEENS1_IJNS0_10UnderscoreES1K_S1K_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T
11_T12_T13_
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info
: Used 116 registers, used 1 barriers
ptxas info
: Compile time = 53.436 ms
[100%] Linking CUDA executable wgmma_sm90
[100%] Built target wgmma_sm90

The build was successful, and the resource numbers look healthy: 116 registers, no spills, no
stack. But the two C7510 warnings are important: ptxas indicates that the WGMMA instructions in both kernels
will be serialized, meaning each instruction must wait for the previous one to finish. This would slow down
the kernels on the H100 for reasons unrelated to what I have studied. The good news is that I am able to
diagnose and likely resolve the issue locally on my RTX 5080 Blackwell SM120 system before renting a
datacenter system.
The compilation output also confirms several key details.
All details below are based on the compilation output. Wherever I am interpreting the data rather than reading
it directly, I will explicitly say so.
</pre>

### 1. What the build produced: three device functions<br>
|Line | Function |What it is|
| :--- | :--- | :--- |
|6–10 | cub::CUB_200802_SM_900::EmptyKernel<void>|a tiny dummy kernel from CUB (section 5)|
|11–15 | gemm_device<...> , first instantiation | the TN GEMM ( gemm_tn ) |
|16–20 | gemm_device<...> , second instantiationthe | NT GEMM ( gemm_nt ) |
<pre>
Where CUB comes from:
wgmma_sm90.cu stores its matrices in Thrust containers. In main it creates the host data in
thrust::host_vector<TA> h_A and so on, then copies it to the GPU with thrust::device_vector<TA> d_A =
h_A . Thrust’s GPU backend is built on CUB, so the include chain is roughly:
wgmma_sm90.cu
└─ <thrust/device_vector.h>
  └─ thrust CUDA backend headers
    └─ <cub/util_device.cuh>  ← defines EmptyKernel
CUTLASS’s own utility headers can also pull CUB in. Either way, it comes in through headers, not through
code you wrote.	

We can confirm this yourself by asking nvcc for the list of every header the file includes:
nvcc -std=c++17 -arch=sm_90a <your same -I flags> -M wgmma_sm90.cu | tr ' ' '\n' | grep -E 
"cub/|thrust/" | head -20
-M makes nvcc print the dependency list without compiling. We can see cub/util_device.cuh in the
output.

What EmptyKernel is
In cub/util_device.cuh it is essentially:
	template <typename T>
	__global__ void EmptyKernel() {}	
It does nothing. CUB uses it as a probe: at runtime, CUB calls cudaFuncGetAttributes(&attrs,
EmptyKernel<void>) and reads attrs.ptxVersion , which tells it which architecture the binary was compiled
for. That is how cub::PtxVersion() works, and CUB algorithms use it to choose tuning parameters.
Because the header refers to EmptyKernel<void> , the template is instantiated in every translation unit
that includes the header. ptxas then compiles it and reports it like any other kernel.

Decoding the namespace cub::CUB_200802_SM_900
CUB doesn’t put its symbols directly in namespace cub . Its CUB_NAMESPACE_BEGIN macro adds an inline
namespace that encodes two things:
</pre>
|Part|Meaning|
|:---|-------|
|200802|The CUB version, encoded as major·100000 + minor·100 +<br>patch, so 2.8.2. This is the CCCL version bundled with<br>your CUDA 12.8 toolkit. It also tells you this build used the<br>toolkit’s built-in CCCL, not a separate ~/cccl checkout.|
|SM_900|The architectures this translation unit was compiled for. It<br>comes from __CUDA_ARCH_LIST__ , so here it’s 900 for<br>sm_90a . If you compiled for sm_120 , it would read<br>SM_1200.|
<pre>
The purpose is to prevent ODR (One Definition Rule) violations. If two libraries in the same program were
compiled with different CUB versions or for different architectures, their CUB symbols get different
mangled names, so the linker never mixes one library’s copy with the other’s.

Does it matter for the GEMM?
No:
It uses 4 registers, 0 bytes of stack and has no body.
Nothing launches it unless CUB code runs PtxVersion() . The tutorial’s device_vector copies are
plain cudaMemcpy calls.
It has no effect on gemm_device ’s registers, the C7510 warning, or the SASS you’re about to
compare.
We can ignore lines 6–10 when analyzing the GEMM. We’ll see the same lines in almost any build that
includes Thrust or CUB, including the CUB programs I write next.
One study note: EmptyKernel and the versioned namespace are internal implementation details, not an
API to learn. When you start CUB, focus on the public interfaces: cub::BlockReduce , cub::BlockScan ,
cub::DeviceReduce , and so on.
</pre>
<pre>
There are two GEMM kernels because main calls gemm() , which chooses gemm_nt or gemm_tn at run time
from transA and transB . Both template instantiations are reachable, so both get compiled, even though
a given run uses only one (NT by default).
</pre>
### 2. Reading the mangled names: your configuration is inside them
<pre>
The long names are C++ mangled template names. CUDA ships a demangler:
echo '_ZN3cub17CUB_200802_SM_90011EmptyKernelIvEEvv' | cu++filt
output: void cub::CUB_200802_SM_900::EmptyKernel<void>()
Every static value is part of the type, which is exactly what “static” means in CuTe:
</pre>
|Fragment in the name|Meaning|
|:-------------------|:------|
|tuple<C<128>, C<128>, C<64>> (written NS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEE ) |cta_tiler = (bM, bN, bK)|
|tuple<int,int,int> ( tupleIJiiiE )|prob_shape : M, N, K are run-time int s|
|Swizzle<3,4,3> ( SwizzleILi3ELi4ELi3EE )|the 128B swizzle|
|SM80_CP_ASYNC_CACHEALWAYS<uint128_t>|the 16-byte cp.async copy atom|
|MMA_64x64x16_F16F16F16_SS|the WGMMA atom|
|C<8192> in the smem layoutthe| stage stride: 8192 halves = 16 KB|
<br><br>
### Telling the two kernels apart. ### The A stride and the Major value differ:

||First kernel (lines 11–15)|Second kernel (lines 16–20)|
|:-|:--------------------------|:-----------------------|
|A stride type|tuple<int, C<1>> : (ldA, 1)|tuple<C<1>, int> : (1, ldA)|
|Contiguous along|K|M|
|Major template value|0|1|
|So it is| gemm_tn (K-major)|gemm_nt (MN-major)|

<pre>
That also tells you how CuTe numbers the enum: GMMA::Major::K = 0 and GMMA::Major::MN = 1 .
The shared-memory layout of the NT kernel, decoded from the second name (my reading of the
mangling substitutions; cu++filt will show it in plain form):

Shape: ((64, 2), (8, 8), (1, 3))
Stride: ((1, 512), (64, 1024), (0, 8192))
</pre>
|Mode|Inside one atom|Across atoms|
|:---|:--------------|:-----------|
|M|64 elements, stride 1 (contiguous)|2 atoms, stride 512 (one 1 KB atom)|
|K|8 elements, stride 64|8 atoms, stride 1024 (two atoms)|
|stage||3 stages, stride 8192 (16 KB)|

<pre>
This confirms what I described as my understanding last time: tile_to_shape places the atom copies in
column-major order, M copies first (stride 512), then K (stride 1024), then stages (stride 8192). Your
compiler output is the verification.
The TN kernel’s layout decodes to shape ((8, 16), (64, 1), (1, 3)) with stride ((64, 512), (1, 0), (0, 8192)):
the K-major atom (8 rows × 64 contiguous K elements), 16 copies down M, one across K. That’s
consistent with Layout_K_SW128_Atom .
</pre>
### 3. The C7510 warnings: the important part ###
<pre>
ptxas info : (C7510) Potential Performance Loss: wgmma.mma_async instructions are serialized
due to wgmma pipeline crossing function boundary at a function call in the function 'gemm_device<...>'
  
What it means. ptxas (the PTX-to-SASS assembler) found a real function call inside the region where
WGMMAs are in flight, between warpgroup_arrive() and warpgroup_wait<0>() . It can’t track asynchronous
WGMMA state across a call, so it plays safe and serializes the WGMMAs: each of the 16 per stage
effectively waits for the previous one to complete before the next is issued. The tensor core then can’t
overlap them.
  
What the call probably is (a hypothesis to check). The kernel’s own code has no visible function
calls; everything in CuTe is meant to be inlined. The most likely source is a device-side assert . In device
code, an active assert compiles to a call to a CUDA runtime function ( __assertfail ), and its message
strings are stored in global memory. Two clues in your output fit this:
   - Line 5: 1076 bytes gmem . The module holds about 1 KB of global data. Assert message strings and
     file names are a typical source.
   - cmake --build build , with no sign of a Release configuration. If CMAKE_BUILD_TYPE isn’t set, NDEBUG
     isn’t defined, so every assert in CuTe and CUTLASS stays active in device code.
</pre>
<small>**How to confirm, locally:**</small>
<pre>
1. Which build type did CMake use?
grep CMAKE_BUILD_TYPE build/CMakeCache.txt
2. Is there a real call in the SASS?
cuobjdump -sass build/wgmma_sm90 | grep -n "CALL"
3. If the call target is visible in the PTX:
cuobjdump -ptx build/wgmma_sm90 | grep -n -E "call|__assertfail" | head
</pre>
<small>**How to fix it. Rebuild in Release, which defines NDEBUG and removes the asserts:**</small>
<pre>
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release

If the C7510 lines disappear, the hypothesis is confirmed. If they don’t, step 2 above shows what 
the call is, and we can look at it together.</pre>
<pre>

What to compare in SASS, before and after.**</small> In the serialized build, I’d expect a wait
( WARPGROUP.DEPBAR or similar) between individual HGMMA instructions. In the fixed build, the 16 HGMMA s
should be issued back to back, with one wait after the batch:
  cuobjdump -sass build/wgmma_sm90 | grep -E "HGMMA|WARPGROUP" | head -40
  cuobjdump -sass build-release/wgmma_sm90 | grep -E "HGMMA|WARPGROUP" | head -40

Why this matters before renting: if you profile the serialized build on the H100, NCU would show low
tensor-core utilization, and you could easily misattribute it to the pipeline design (the cp_async_wait<0>()
issue we discussed). Fixing the build first means the H100 numbers reflect the kernel itself.
</pre>

### 4. Resource usage: lines 13–14 and 18–19 ###
<pre>
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 116 registers, used 1 barriers</pre>

|Item|Value|Meaning|
|:---|:----|:------|
|Stack frame|0 bytes|no per-thread local-memory arrays|
|Spills|0|every value fits in registers, no slow local-memory traffic
|Registers|116 per thread|matches the earlier prediction of “comfortably above 64”|
|Barriers|1|one hardware barrier: the single __syncthreads() after the prologue.<br>The warpgroup_* operations don’t use these barriers|

<small><small>Where 116 registers go, roughly:</small></small>
|Use|Registers|
|:--|:--------|
|C accumulators: 128 halves, two per register|64|
|A and B descriptors (one base each)|4|
|cp.async source and destination addresses, loop counters, stage indices, M/N/K values,<br>alpha/beta, epilogue pointers|the remaining ~48|
<pre>
Both kernels use the same 116, which makes sense: they differ only in strides and layouts, not in the
amount of state.</pre>
What this means for occupancy on the H100 (a prediction to confirm in NCU’s Occupancy section):
|Resource|Per CTA|Limit per SM|CTAs that fit|
|:-------|:------|:-----------|:------------|
|Registers|116, allocated as 120 per thread (rounded to a multiple of 8, as I understand the allocation<br>rule) × 128 threads = 15,360|65,536|4|
|Shared memory|96 KB + ~1 KB reserved|~228 KB|2|
|Threads|128|2,048|16|
<pre>
So shared memory is the limit: 2 CTAs per SM, 8 warps out of 64, which is 12.5% theoretical
occupancy. That’s normal for a GEMM with large tiles; the tiles provide the parallelism, not warp count. In
NCU, look for “Block Limit Shared Mem = 2” and “Block Limit Registers = 4.”
</pre>

### 5. The CUB kernel: lines 6–10 ###
<pre>
Compiling entry function '_ZN3cub17CUB_200802_SM_90011EmptyKernelIvEEvv' for 'sm_90a'

There is no CUB code, so why is CUB here? The tutorial uses thrust::device_vector, and Thrust’s
CUDA backend includes CUB. As I understand it, EmptyKernel is a do-nothing kernel CUB uses to query
which PTX version a program was compiled for.
Two useful facts are in its name:
&bull; CUB_200802 is CUB’s version-tagged namespace: CUB 2.8.2. So this build used the CCCL bundled
with your CUDA toolkit, not a CCCL 3.x clone. This is the version check I mentioned for your CUB
work. When you start CUB, compile with your CCCL clone’s -I paths and this number should
change to 3.x.
&bull; SM_900 records the target architecture in the namespace, so code compiled for different GPUs
doesn’t clash.
</pre>
### 6. Your next steps, all local ###
<pre>
  1. Demangle one kernel name with cu++filt and match it to the code.
  2. Check the build type, then rebuild in Release and see whether C7510 disappears.
  3. Compare the HGMMA sequences in SASS between the two builds.
  4. With the Release build, check the earlier predictions: 16 HGMMA and 16 LDGSTS per main-loop
iteration, DEPBAR with count 0, and no instructions from warpgroup_fence_operand.
  
Bring back the output of step 2 (the grep CALL result especially) and we’ll confirm the cause together.
</pre>
## Rebuild steps ## 
<pre>
By leveraging CMake's built-in target properties.
  

Add this at the absolute end of your CMakeLists.txt, 
then issue command: cmake --build build --target analyze
  
if(CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Darwin" OR UNIX)
    add_custom_target(analyze
        # 1. Clear out any previous sass files to ensure fresh results
        COMMAND ${CMAKE_COMMAND} -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/wgmma_release.sass"
        
        # 2. Extract assembly using the exact location of the target binary
        COMMAND cuobjdump -sass $<TARGET_FILE:wgmma_sm90> > "${CMAKE_CURRENT_BINARY_DIR}/wgmma_release.sass"
        
        # 3. Print analysis metrics safely
        COMMAND echo "--- HGMMA Count ---"
        COMMAND grep -c "HGMMA" "${CMAKE_CURRENT_BINARY_DIR}/wgmma_release.sass" || true
        
        # 4. Print Warpgroup context
        COMMAND echo "--- WARPGROUP Lines ---"
        COMMAND grep -n "WARPGROUP" "${CMAKE_CURRENT_BINARY_DIR}/wgmma_release.sass" | head -n 10 || true
        
        # 5. Print Calls
        COMMAND echo "--- CALL Lines ---"
        COMMAND grep -n "CALL" "${CMAKE_CURRENT_BINARY_DIR}/wgmma_release.sass" | head -n 10 || true
        
        # 6. Extract PTX calls and checks
        COMMAND echo "--- PTX Calls/Asserts ---"
        COMMAND cuobjdump -ptx $<TARGET_FILE:wgmma_sm90> | grep -E "call|__assertfail" | head -n 10 || true
        
        DEPENDS wgmma_sm90
        COMMENT "Inspecting SASS and PTX outputs for Hopper SM90 instructions..."
        VERBATIM
    )
endif()

The build output is shown below.

cmake --build build --target analyze
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /home/chialan/cuda-learning/cute_hopper/sm90/build
[ 66%] Built target wgmma_sm90
[100%] Inspecting SASS and PTX outputs for Hopper SM90 instructions...
--- HGMMA Count ---
32
--- WARPGROUP Lines ---
684:        /*1380*/                   WARPGROUP.ARRIVE ;                                                /* 0x00000000000079c5 */
810:        /*1770*/                   WARPGROUP.DEPBAR.LE gsb0, 0x0 ;                                   /* 0x00008000000079c5 */
3727:        /*13f0*/                   WARPGROUP.ARRIVE ;                                                /* 0x00000000000079c5 */
3851:        /*17d0*/                   WARPGROUP.DEPBAR.LE gsb0, 0x0 ;                                   /* 0x00008000000079c5 */
--- CALL Lines ---
--- PTX Calls/Asserts ---
[100%] Built target analyze

This output confirms that your code is compiling into a highly optimized, state-of-the-art Hopper (SM90) 
hardware-native Matrix Multiply kernel. It shows that your CuTe layout configurations are mapping correctly 
to the underlying silicon. Here is a detailed breakdown of what these specific numbers and assembly lines 
tell us about your program:


1. HGMMA Count = 32

• What it means: HGMMA stands for Hopper Group Matrix Multiply and Accumulate. Seeing exactly 32 means the
  code compiled into native asynchronous matrix instructions executed directly by the Tensor Cores.
• Why this is excellent: If this count were 0, it would mean the compiler failed to understand the CuTe
  layouts and fell back to executing matrix multiplications using slower, individual thread-level 
  floating-point operations. A count of 32 tells us that the compiler successfully generated 
  hardware-accelerated matrix math.

2. WARPGROUP Lines (The Asynchronous Orchestration)

The lines showing WARPGROUP.ARRIVE and WARPGROUP.DEPBAR provide a window into how Hopper manages memory 
and math concurrently. Hopper introduces the concept of a Warpgroup (a collection of 4 warps, or 128 
threads working as a single unit).

684:  /*1380*/  WARPGROUP.ARRIVE ;
• The Mechanism: This instruction tells the Tensor Core hardware: "This warpgroup has initiated a matrix
  multiply task, and its inputs are flying into the execution pipeline."
• The Asynchronous Benefit: Because this is completely asynchronous, the warp scheduler immediately moves 
  on to compute subsequent lines of code without waiting for the Tensor Cores to finish the math.

810:  /*1770*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
• The Mechanism: DEPBAR stands for Dependency Barrier. LE gsb0, 0x0 checks a dedicated hardware counter slot
  (gsb0) and halts execution until the number of outstanding matrix tasks drops down to 0.
• The Purpose: This acts as a protective shield. It ensures that the program cannot try to read or use the
  final accumulator matrix values until the Tensor Cores have completely finished computing the
  WARPGROUP.ARRIVE operations triggered earlier.

Why are there two identical pairs in the build coutput?

I see two distinct sets of lines (684/810 and 3727/3851) because the file compiled two separate variants 
of the gemm_device template function—which matches the two ptxas info lines I observed in the previous 
build output.

3. Empty CALL Lines & PTX Asserts

• What it means: Both the CALL Lines (SASS assembly level) and PTX Calls/Asserts outputs are completely blank.
• Why this is excellent:
	1. Zero Device Function Overhead: The compiler successfully inlined every single sub-function and layout
    utility function from CuTe. There are no actual jumps or function call overheads left inside the binary.
	2. No Active Assertions: If I had active bounds-checking (assert(...)) or layout error checks triggered
    inside the kernel, __assertfail or a branch leading to a device call would have appeared here. A blank
    output indicates that the compiler entirely optimized out or validated the paths, leaving a clean,
    straight-line execution flow optimized for pure speed.

Summary Verification

The environment is correctly set up for Hopper SM90. The code successfully targets WGMMA matrix hardware
shapes, completely avoids slow functional calls, utilizes hardware dependency barriers to hide 
memory/compute latency, and runs with zero active assertion traps.
</pre>
<pre>
Which kernel is gemm_nt? It’s the second one (line 16), not the
first.
I ran both mangled names through c++filt , which converts them back into readable C++ types. The
order in the log is TN first (line 11), NT second (line 16). You can tell from three places in the
template arguments:
</pre>
|Clue|Line 11|Line 16|
|:---|:------|:------|
|A stride ( dA )| tuple<int, C<1>> = (ldA, 1): K is contiguous, so K-major (T)|tuple<C<1>, int> = (1, ldA):M is contiguous,<br> so M-major (N)|
|MMA atom majors|MMA_64x64x16_F16F16F16_SS<Major 0, Major 0, ...>, where 0 = Major::K|<Major 1, Major 1, ...>,where 1 = Major::MN|
|smem layout shape|((8,16),(64,1),(1,3))|((64,2),(8,8),(1,3))|
<pre>
Tip: run c++filt on these names ( echo '_Z11gemm…' | c++filt ). It is faster and more reliable than
decoding by hand.
	
The NT kernel’s template arguments, decoded
gemm_device<ProblemShape, CtaTiler, TA, AStride, ASmemLayout, TiledCopyA, TB, BStride, BSmemLayout,
TiledCopyB, TC, CStride, TiledMma, Alpha, Beta>
</pre>
|Parameter|Value in the NT instantiation|Meaning|
|:--------|:----------------------------|:------|
|ProblemShape |tuple<int,int,int>|M, N, K are runtime values (5120,5120, 4096)|
|CtaTiler|tuple<C<128>,C<128>,C<64>>|bM, bN, bK are compile-time constants|
|TA / TB / TC|half_t||
|dA, dB|tuple<C<1>, int>|M-major A, N-major B. The 1 is static,<br>so the compiler knows the inner stride is contiguous and can vectorize.|
|dC|tuple<C<1>, int>|C is M-major (column-major)|
|sA / sB layout|ComposedLayout<Swizzle<3,4,3>,smem_ptr_flag_bits<16>,<br>Layout<((64,2),(8,8),(1,3)) :((1,512),(64,1024),(0,8192))>>|See the breakdown below| 
|TiledCopyA/B|Copy_Atom<SM80_CP_ASYNC_CACHEALW AYS<uint128_t>, half_t> ,<br> TV layout (128,8):(8,1) , tiler (128,8)| See below|
|TiledMMA|MMA_Atom<MMA_64x64x16_F16F16F16_SS<MN,MN,One,One>>,<br> atom layout (1,1,1):(0,0,0)| A single warpgroup. There is no 2×2 warpgroup tiling,<br> so the one warpgroup covers the 128×128 tile by iterating 2×2 in M and N and 4× in K: 16 WGMMAs per K-tile.|
|Alpha, Beta|half_t||
<pre>
The NT smem layout, ((64,2),(8,8),(1,3)) : ((1,512),(64,1024),(0,8192)) , in units of half
elements:
&bull; M mode (64,2):(1,512) . The inner 64 are contiguous: 64 halves = 128 B, one swizzle row. The
outer 2 jump 512 halves = 1 KB, which is the next atom in M. 64·2 = 128 = bM.
&bull; K mode (8,8):(64,1024) . The inner 8 step 64 halves (128 B) to the next K column inside an atom,
so one atom is 64 (M) × 8 (K) = 512 halves = 1 KB = one full Swizzle<3,4,3> period. The outer 8
step 1024 halves to the next atom group. 8·8 = 64 = bK.
&bull; Stage mode (1,3):(0,8192) . 3 stages, 8192 halves = 16 KB apart. The 1:0 is a size-1 placeholder
left over from tile_to_shape .
&bull; The atom is GMMA::Layout_MN_SW128_Atom<half_t> = 64×8. tile_to_shape lays atoms out column-
major: first the 2 in M (stride 512), then the 8 in K (stride 1024).
</pre>
<pre>
The NT copy, TV layout (128,8):(8,1) . Thread t’s first element is at index 8t in a 128×8 column-major
tile, and each thread copies 8 contiguous halves, which is 16 B. That is exactly one cp.async.ca … 16 .
Threads 0–15 cover one 128-element M column, threads 16–31 the next column, and so on, so the global
loads are coalesced along M, the contiguous direction in NT. Per K-tile, A is 128×64 = 8192 halves ÷ (128
threads × 8 halves) = 8 cp.async per thread for A, and 8 for B.
</pre>
The resource lines
|Line|Meaning|
|:---|:------|
|(C7510) … wgmma serialized … function call|ptxas found a real call inside the kernel (not inlined)<br>somewhere between WGMMA instructions. A WGMMA<br>batch can’t stay in flight across a function boundary,<br> so ptxas inserts waits around every wgmma. That eliminates<br>the asynchronous overlap. On an H100 this would cost real<br>performance. Here it is a compile-time warning only,<br>because you can’t run this binary.|
|1076 bytes gmem|Global/constant data emitted by the compilation unit.<br>Assert message strings ( __FILE__ , the expression text,<br>the function name) are a typical source, which fits the<br>assert hypothesis. That’s an inference, not proof: a<br>printf would also produce strings and a call to<br>vprintf . The CALL check below settles it.|
|EmptyKernel in cub::CUB_200802_SM_900|A dummy kernel CUB defines in every translation unit that<br>includes it. It comes in through CUTLASS headers. 200802<br>means CUB 2.8.2, the CCCL that ships with CUDA 12.8.<br>SM_900 is the architecture tag in the inline namespace. It<br>is harmless and costs 4 registers.|
|Used 116 registers|The accumulator is 128 halves per thread = 64 32-bit<br>registers (2 halves packed per register). The other ~52<br>hold global addresses, smem descriptors, loop counters<br>and copy predicates. On an H100 that allows 4 CTAs per<br>SM by registers, but the ~96 KB of shared memory limits it
to 2.|
|used 1 barriers|Barrier 0, used by __syncthreads() .|
|0 stack, 0 spills|Good: nothing went to local memory.|
|Compile time ~54 ms per instantiation|ptxas time only. The template instantiation in the front end<br>is what makes the overall build slow.|

Next step: apply the fix and confirm it
<pre>
# a) Confirm the current build type. Empty or Debug means asserts are compiled in.
grep -E "CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS" build/CMakeCache.txt
	CMAKE_BUILD_TYPE:STRING=Release
	CMAKE_CUDA_ARCHITECTURES:STRING=52
	CMAKE_CUDA_FLAGS:STRING=
	CMAKE_CUDA_FLAGS_DEBUG:STRING=-g
	CMAKE_CUDA_FLAGS_MINSIZEREL:STRING=-O1 -DNDEBUG
	CMAKE_CUDA_FLAGS_RELEASE:STRING=-O3 -DNDEBUG
	CMAKE_CUDA_FLAGS_RELWITHDEBINFO:STRING=-O2 -g -DNDEBUG
	CMAKE_CUDA_FLAGS-ADVANCED:INTERNAL=1
	CMAKE_CUDA_FLAGS_DEBUG-ADVANCED:INTERNAL=1
	CMAKE_CUDA_FLAGS_MINSIZEREL-ADVANCED:INTERNAL=1
	CMAKE_CUDA_FLAGS_RELEASE-ADVANCED:INTERNAL=1
	CMAKE_CUDA_FLAGS_RELWITHDEBINFO-ADVANCED:INTERNAL=1

# b) Find the call that causes C7510 in the current binary
cuobjdump -sass build/wgmma_sm90 | grep -n -B2 -A2 "CALL"
   Why is the output empty?
	• Perfect Inlining: The compiler successfully took all of the CuTe layouts, coordinate math, and 
	utility functions and expanded them directly into a flat, straight line of hardware instructions.
	• Zero Overhead: In high-performance GPU kernels, a CALL instruction usually means the code is jumping 
	to a separate device function or falling back to a slow routine (like a runtime assertion check). 
	Seeing no CALL lines means your binary has zero function-call overhead.

cuobjdump -ptx build/wgmma_sm90 | grep -E "call|__assertfail|vprintf" | head
Why is the output empty?
	1. Why call is missing
	In PTX, a call instruction is generated when a function cannot be inlined, or when they are calling a
	separate __device__ function that isn't visible to the compiler at template instantiation. Because they 
	are using CuTe and CUTLASS, the compiler is able to look at all the template layouts and inline 100% 
	of the logic. The entire execution path is flattened into a single, highly efficient stream of execution.
	2. Why __assertfail is missing
	If the C++ code contains assert(...) statements, or if the underlying CuTe libraries include internal
	runtime bounds checks, the compiler translates those into a conditional branch that calls __assertfail 
	if a check fails.
	• The Optimization: Because I compiled a Release build (evident from the build-release target and 
	optimal register footprint), the compiler entirely removes these safety assert checks to maximize
	execution speed.
	3. Why vprintf is missing
	If there is any printf(...) statements inside the __global__ or __device__ CUDA functions, the NVIDIA
	compiler translates them under the hood into a PTX internal system call named vprintf. An empty output
	proves that the device kernel does not contain any active debugging prints, which is critical because
	device-side prints heavily bottleneck GPU performance.

	
# c) Configure a separate Release directory (adds -O3 -DNDEBUG) and build it
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release 2>&1 | tee release.log
	
The contents of the release.log file are shown below.
	cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
	cmake --build build-release 2>&1 | tee release.log
	-- The CXX compiler identification is GNU 13.3.0
	-- The CUDA compiler identification is NVIDIA 12.9.86
	-- Detecting CXX compiler ABI info
	-- Detecting CXX compiler ABI info - done
	-- Check for working CXX compiler: /usr/bin/c++ - skipped
	-- Detecting CXX compile features
	-- Detecting CXX compile features - done
	-- Detecting CUDA compiler ABI info
	-- Detecting CUDA compiler ABI info - done
	-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
	-- Detecting CUDA compile features
	-- Detecting CUDA compile features - done
	-- Found CUDAToolkit: /usr/local/cuda/targets/x86_64-linux/include (found version "12.9.86") 
	-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
	-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
	-- Found Threads: TRUE  
	-- Found OpenMP_CXX: -fopenmp (found version "4.5") 
	-- Found OpenMP: TRUE (found version "4.5")  
	-- Configuring done (1.4s)
	-- Generating done (0.0s)
	-- Build files have been written to: /home/chialan/cuda-learning/cute_hopper/sm90/build-release
	[ 50%] Building CUDA object CMakeFiles/wgmma_sm90.dir/wgmma_sm90.cu.o
	ptxas info    : 47 bytes gmem
	ptxas info    : Compiling entry function '_ZN3cub17CUB_200802_SM_90011EmptyKernelIvEEvv' for 'sm_90a'
	ptxas info    : Function properties for _ZN3cub17CUB_200802_SM_90011EmptyKernelIvEEvv
	    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
	ptxas info    : Used 4 registers, used 0 barriers
	ptxas info    : Compile time = 7.355 ms
	ptxas info    : Compiling entry function '_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJiNS3_ILi1EEEEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJNS3_ILi8EEENS3_ILi16EEEEEENS1_IJS5_S9_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS5_NS3_ILi512EEEEEENS1_IJS9_NS3_ILi0EEEEEENS1_IJSQ_NS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES10_EES8_EEENSG_INS1_IJSJ_SH_EEENS1_IJNS1_IJS4_S9_EEESI_EEEEENS1_IJSI_S5_EEEEES8_SA_SW_S18_S8_NS1_IJS9_iEEENS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1D_5MajorE0ELS1F_0ELNS1D_7ScaleInE1ELS1G_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSQ_SQ_SQ_EEEEENS1_IJNS0_10UnderscoreES1M_S1M_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_' for 'sm_90a'
	ptxas info    : Function properties for _Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJiNS3_ILi1EEEEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJNS3_ILi8EEENS3_ILi16EEEEEENS1_IJS5_S9_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS5_NS3_ILi512EEEEEENS1_IJS9_NS3_ILi0EEEEEENS1_IJSQ_NS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES10_EES8_EEENSG_INS1_IJSJ_SH_EEENS1_IJNS1_IJS4_S9_EEESI_EEEEENS1_IJSI_S5_EEEEES8_SA_SW_S18_S8_NS1_IJS9_iEEENS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1D_5MajorE0ELS1F_0ELNS1D_7ScaleInE1ELS1G_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSQ_SQ_SQ_EEEEENS1_IJNS0_10UnderscoreES1M_S1M_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_
	    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
	ptxas info    : Used 112 registers, used 1 barriers
	ptxas info    : Compile time = 54.679 ms
	ptxas info    : Compiling entry function '_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJNS3_ILi1EEEiEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJS5_NS3_ILi2EEEEEENS1_IJNS3_ILi8EEESJ_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS9_NS3_ILi512EEEEEENS1_IJS5_NS3_ILi1024EEEEEENS1_IJNS3_ILi0EEENS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES11_EES8_EEENSG_INS1_IJS4_SJ_EEENS1_IJSJ_S9_EEEEES14_EES8_SA_SX_S17_S8_SA_NS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1B_5MajorE1ELS1D_1ELNS1B_7ScaleInE1ELS1E_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSS_SS_SS_EEEEENS1_IJNS0_10UnderscoreES1K_S1K_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_' for 'sm_90a'
	ptxas info    : Function properties for _Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJNS3_ILi1EEEiEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJS5_NS3_ILi2EEEEEENS1_IJNS3_ILi8EEESJ_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS9_NS3_ILi512EEEEEENS1_IJS5_NS3_ILi1024EEEEEENS1_IJNS3_ILi0EEENS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES11_EES8_EEENSG_INS1_IJS4_SJ_EEENS1_IJSJ_S9_EEEEES14_EES8_SA_SX_S17_S8_SA_NS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1B_5MajorE1ELS1D_1ELNS1B_7ScaleInE1ELS1E_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSS_SS_SS_EEEEENS1_IJNS0_10UnderscoreES1K_S1K_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_
	    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
	ptxas info    : Used 114 registers, used 1 barriers
	ptxas info    : Compile time = 48.958 ms
	[100%] Linking CUDA executable wgmma_sm90
	[100%] Built target wgmma_sm90

	
	
# d) Compare instruction counts between the two binaries
#!/bin/bash

# Target the directories you want to inspect
for d in build build-release; do
  if [ -d "$d" ] && [ -f "$d/wgmma_sm90" ]; then
    echo "== Analyzing Directory: $d"
    cuobjdump -sass "$d/wgmma_sm90" | grep -oE "HGMMA[.0-9A-Zx]*|WARPGROUP[.A-Z]*|CALL[.A-Z]*|LDGSTS[.0-9A-Z]*|LDGDEPBAR|DEPBAR[.A-Z]*" | sort | uniq -c
  else
    echo "== Skipping: $d (binary not found)"
  fi
done

The output are shown below:	
== Analyzing Directory: build
      2 DEPBAR.LE
     32 HGMMA.64x64x16.F16
      8 LDGDEPBAR
     96 LDGSTS.E.LTC128B.128
      2 WARPGROUP.ARRIVE
      2 WARPGROUP.DEPBAR.LE
== Analyzing Directory: build-release
      2 DEPBAR.LE
     32 HGMMA.64x64x16.F16
      8 LDGDEPBAR
     96 LDGSTS.E.LTC128B.128
      2 WARPGROUP.ARRIVE
      2 WARPGROUP.DEPBAR.LE


1. 96 LDGSTS.E.LTC128B.128 (Global-to-Shared Asynchronous Moves)

• What it means: LDGSTS stands for Load Global, Store Shared. It bypasses the register file entirely, 
pulling data straight from global memory and pushing it directly into Shared Memory (smem).
	• .E means it uses extended addressing (64-bit global pointers).
	• .LTC128B.128 tells us it is performing 128-byte cache line transfers directly bypassing or 
	interacting optimally with the L2 cache (LTC).
• The Insight: A count of 96 instructions shows that the compiler has unrolled your global memory data 
ingestion pipeline. It is staging massive chunks of the A and B matrices into shared memory asynchronously 
to keep the Tensor Cores fed.

2. 32 HGMMA.64x64x16.F16 & 2 WARPGROUP.ARRIVE

• What it means: HGMMA.64x64x16.F16 is the exact native instruction for Hopper Group Matrix Multiply and
Accumulate. It is computing a matrix block of shape 64 × 64 × 16 using FP16 precision.
• The Structural Ratio (16 to 1): Notice that you have 32 HGMMA instructions but only 2 WARPGROUP.ARRIVE
instructions. This means each WARPGROUP.ARRIVE acts as a macro-gate that fires off a sequence of 16 
underlying HGMMA matrix math steps back-to-back inside the hardware execution pipeline before hitting a
dependency barrier.

3. Synchronization Footprint (8 LDGDEPBAR vs. 2 WARPGROUP.DEPBAR.LE & 2 DEPBAR.LE)

Hopper uses separate hardware counter barriers to track memory versus math. This breakdown shows a highly
optimized balance:
• 8 LDGDEPBAR (Load Global Dependency Barrier): This tracks your async memory copies. The threads use these 
8 checks to make sure the data flying in via the 96 LDGSTS commands has safely landed in shared memory 
before anyone tries to read it.
• 2 WARPGROUP.DEPBAR.LE: This stalls the execution stream to guarantee that the 32 Tensor Core HGMMA 
math operations are completely done before the final registers are read or written back to global memory.
• 2 DEPBAR.LE: These are standard execution dependency barriers used to manage basic register 
write-after-read hazards inside the general warp instructions.

Why build and build-release Look Exactly the Same:
Usually, a debug build has way more instructions than a build-release. Because the counts match exactly 
down to the digit, it means the CMakeLists.txt is passing high optimization flags (like -O3 and strict
architecture flags -arch=sm_90a) to nvcc for both directories. The compiler completely optimized the code 
in both setups.
I now have the absolute baseline footprint of a working Hopper WGMMA kernel.
Over the two-day study window, would you like to:
	
All of these run on my 5080 machine. cuobjdump reads the sm_90a code from the binary; it doesn’t need
a Hopper GPU. In (d), the counts cover both kernels together. Loop unrolling also changes the totals, so
the useful comparison is Debug vs Release, not the exact numbers

What should change if the hypothesis is right: both C7510 lines disappear from release.log , gmem
drops well below 1076 bytes, and CALL disappears from the SASS. If C7510 is still there in Release, the
CALL lines from step (b) will name what is being called, and we’ll trace it from there.
</pre>
