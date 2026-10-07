
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
</pre>
