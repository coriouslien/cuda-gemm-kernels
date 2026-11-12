<pre>
The following is the compiling output of wgmma_sm90.cu in my desktop RTX 5080 Blackwell SM120, not Lambda.
make --build build
/usr/bin/cmake -S/home/xxx/cuda-learning/cute_hopper/lambda_sm90 -B/home/xxx/cuda-learning/cute_hopper/lambda_sm90/build --check-build-system CMakeFiles/Makefile.cmake 0
/usr/bin/cmake -E cmake_progress_start /home/xxx/cuda-learning/cute_hopper/lambda_sm90/build/CMakeFiles /home/xxx/cuda-learning/cute_hopper/lambda_sm90/build//CMakeFiles8./progress.marks
/usr/bin/gmake  -f CMakeFiles/Makefile2 all
gmake[1]: Entering directory '/home/xxx/cuda-learning/cute_hopper/lambda_sm90/build'
/usr/bin/gmake  -f CMakeFiles/wgmma_sm90.dir/build.make CMakeFiles/wgmma_sm90.dir/depend
gmake[2]: Entering directory '/home/xxx/cuda-learning/cute_hopper/lambda_sm90/build'
cd /home/xxx/cuda-learning/cute_hopper/lambda_sm90/build && /usr/bin/cmake -E cmake_depends "Unix Makefiles" /home/xxx/cuda-learning/cute_hopper/lambda_sm90 /home/xxx/cuda-learning/cute_hopper/lambda_sm90 /home/xxx/cuda-learning/cute_hopper/lambda_sm90/build /home/xxx/cuda-learning/cute_hopper/lambda_sm90/build /home/xxx/cuda-learning/cute_hopper/lambda_sm90/build/CMakeFiles/wgmma_sm90.dir/DependInfo.cmake "--color="
gmake[2]: Leaving directory '/home/xxx/cuda-learning/cute_hopper/lambda_sm90/build'
/usr/bin/gmake  -f CMakeFiles/wgmma_sm90.dir/build.make CMakeFiles/wgmma_sm90.dir/build
gmake[2]: Entering directory '/home/xxx/cuda-learning/cute_hopper/lambda_sm90/build'
[ 50%] Building CUDA object CMakeFiles/wgmma_sm90.dir/wgmma_sm90.cu.o
/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler  --options-file CMakeFiles/wgmma_sm90.dir/includes_CUDA.rsp -O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_90a,code=[compute_90a,sm_90a]" --expt-relaxed-constexpr --generate-line-info -Xptxas=-v -Xptxas=-warn-spills --resource-usage -UCUTLASS_ENABLE_SYNCLOG -MD -MT CMakeFiles/wgmma_sm90.dir/wgmma_sm90.cu.o -MF CMakeFiles/wgmma_sm90.dir/wgmma_sm90.cu.o.d -x cu -c /home/xxx/cuda-learning/cute_hopper/lambda_sm90/wgmma_sm90.cu -o CMakeFiles/wgmma_sm90.dir/wgmma_sm90.cu.o
ptxas info    : 47 bytes gmem
ptxas info    : Compiling entry function '_ZN3cub17CUB_200802_SM_90011EmptyKernelIvEEvv' for 'sm_90a'
ptxas info    : Function properties for _ZN3cub17CUB_200802_SM_90011EmptyKernelIvEEvv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 4 registers, used 0 barriers
ptxas info    : Compile time = 7.958 ms
ptxas info    : Compiling entry function '_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJiNS3_ILi1EEEEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJNS3_ILi8EEENS3_ILi16EEEEEENS1_IJS5_S9_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS5_NS3_ILi512EEEEEENS1_IJS9_NS3_ILi0EEEEEENS1_IJSQ_NS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES10_EES8_EEENSG_INS1_IJSJ_SH_EEENS1_IJNS1_IJS4_S9_EEESI_EEEEENS1_IJSI_S5_EEEEES8_SA_SW_S18_S8_NS1_IJS9_iEEENS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1D_5MajorE0ELS1F_0ELNS1D_7ScaleInE1ELS1G_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSQ_SQ_SQ_EEEEENS1_IJNS0_10UnderscoreES1M_S1M_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_' for 'sm_90a'
ptxas info    : Function properties for _Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJiNS3_ILi1EEEEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJNS3_ILi8EEENS3_ILi16EEEEEENS1_IJS5_S9_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS5_NS3_ILi512EEEEEENS1_IJS9_NS3_ILi0EEEEEENS1_IJSQ_NS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES10_EES8_EEENSG_INS1_IJSJ_SH_EEENS1_IJNS1_IJS4_S9_EEESI_EEEEENS1_IJSI_S5_EEEEES8_SA_SW_S18_S8_NS1_IJS9_iEEENS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1D_5MajorE0ELS1F_0ELNS1D_7ScaleInE1ELS1G_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSQ_SQ_SQ_EEEEENS1_IJNS0_10UnderscoreES1M_S1M_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 112 registers, used 1 barriers
ptxas info    : Compile time = 57.885 ms
ptxas info    : Compiling entry function '_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJNS3_ILi1EEEiEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJS5_NS3_ILi2EEEEEENS1_IJNS3_ILi8EEESJ_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS9_NS3_ILi512EEEEEENS1_IJS5_NS3_ILi1024EEEEEENS1_IJNS3_ILi0EEENS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES11_EES8_EEENSG_INS1_IJS4_SJ_EEENS1_IJSJ_S9_EEEEES14_EES8_SA_SX_S17_S8_SA_NS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1B_5MajorE1ELS1D_1ELNS1B_7ScaleInE1ELS1E_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSS_SS_SS_EEEEENS1_IJNS0_10UnderscoreES1K_S1K_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_' for 'sm_90a'
ptxas info    : Function properties for _Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJNS3_ILi1EEEiEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJS5_NS3_ILi2EEEEEENS1_IJNS3_ILi8EEESJ_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS9_NS3_ILi512EEEEEENS1_IJS5_NS3_ILi1024EEEEEENS1_IJNS3_ILi0EEENS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES11_EES8_EEENSG_INS1_IJS4_SJ_EEENS1_IJSJ_S9_EEEEES14_EES8_SA_SX_S17_S8_SA_NS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1B_5MajorE1ELS1D_1ELNS1B_7ScaleInE1ELS1E_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSS_SS_SS_EEEEENS1_IJNS0_10UnderscoreES1K_S1K_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 114 registers, used 1 barriers
ptxas info    : Compile time = 51.020 ms
[100%] Linking CUDA executable wgmma_sm90
/usr/bin/cmake -E cmake_link_script CMakeFiles/wgmma_sm90.dir/link.txt --verbose=1
/usr/bin/g++ @CMakeFiles/wgmma_sm90.dir/objects1.rsp -o wgmma_sm90 @CMakeFiles/wgmma_sm90.dir/linkLibs.rsp -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" -L"/usr/local/cuda/targets/x86_64-linux/lib"
gmake[2]: Leaving directory '/home/xxx/cuda-learning/cute_hopper/lambda_sm90/build'
[100%] Built target wgmma_sm90
gmake[1]: Leaving directory '/home/xxx/cuda-learning/cute_hopper/lambda_sm90/build'
/usr/bin/cmake -E cmake_progress_start /home/xxx/cuda-learning/cute_hopper/lambda_sm90/build/CMakeFiles 0


Detailed analysis of the wgmma_sm90.cu compilation output
The compilation output confirms that I successfully built Hopper-specific CUDA machine code
(sm_90a) on my RTX 5080 Blackwell SM120 desktop.

The most interesting discovery is that the compiler generated two specialized GEMM kernels,
using 112 and 114 registers per thread. The earlier Nsight Compute report from the H100 showed
114 registers per thread, so I can connect the compilation output directly to the kernel I
profiled.

Explaning the build process first, then the compiler flags, the two GEMM specializations,
register usage, and finally their relationship to the H100 performance results.
1. Understand the compilation process
The build follows this sequence:
CMake + GNU Make
Check dependencies and determine which files require rebuilding
NVIDIA CUDA compiler — nvcc
Compile CUDA C++ templates for compute_90a and sm_90a
PTX assembler — ptxas
Generate Hopper machine code; allocate registers and barrier resources
Host linker — g++
Link objects and required libraries into one executable
build/wgmma_sm90
Executable containing the host application and Hopper device code
The stages are conceptually separate, although nvcc orchestrates several of them internally.

My desktop RTX 5080 Blackwell SM120 does not have to execute the Hopper instructions during
compilation. The compiler only needs to support their instruction set and architecture target.
This is why my approach of compiling locally and executing on Lambda H100 works.
    
2. The first part: CMake and GNU Make
The log begins with commands such as:
/usr/bin/cmake -S... -B... --check-build-system ...
This tells CMake to check whether the generated build system is still valid.
Next:
/usr/bin/gmake -f CMakeFiles/Makefile2 all
This invokes GNU Make to build the default target.
The nested gmake[1] and gmake[2] messages reflect recursively invoked Make processes. They do
not indicate two different GPUs or two compiler executions.

The build then reaches:
[ 50%] Building CUDA object CMakeFiles/wgmma_sm90.dir/wgmma_sm90.cu.o
This means CMake is compiling the source file into an object file:
wgmma_sm90.cu
       |
       V
wgmma_sm90.cu.o

The object file is not yet the final executable. It contains compiled code and information 
needed by the linker.
The percentage is CMake's build progress, not an indication that half of the CUDA kernel is 
executing.

3. Understand the exact nvcc command
The central command is:
/usr/local/cuda/bin/nvcc \
    -forward-unknown-to-host-compiler \
    --options-file CMakeFiles/wgmma_sm90.dir/includes_CUDA.rsp \
    -O3 -DNDEBUG -std=c++17 \
    "--generate-code=arch=compute_90a,code=[compute_90a,sm_90a]" \
    --expt-relaxed-constexpr \
    --generate-line-info \
    -Xptxas=-v \
    -Xptxas=-warn-spills \
    --resource-usage \
    -UCUTLASS_ENABLE_SYNCLOG \
    -MD \
    -MT CMakeFiles/wgmma_sm90.dir/wgmma_sm90.cu.o \
    -MF CMakeFiles/wgmma_sm90.dir/wgmma_sm90.cu.o.d \
    -x cu -c wgmma_sm90.cu \
    -o CMakeFiles/wgmma_sm90.dir/wgmma_sm90.cu.o

It has been shortened the absolute source path for readability. The following explanation
covers each group of options.
3.1 Optimization and language options
</pre>
|Option| Meaning|Why it matters|
|:-----|:-------|:-------------|
|-O3|Enable aggressive host compiler optimization|Appropriate for performance experiments|
|-DNDEBUG|Define the NDEBUG macro|Disables standard C/C++ assert() checks|
|-std=c++17|Compile using C++17|Required by this CUTLASS/CuTe build|
|--expt-relaxed-constexpr|Relax certain host/device<br> constexpr restrictions|	Useful for<br> 
template-heavy CUDA C++|
|-forward-unknown-to-host-compiler|Forward qualifying unknown<br> options to the host compiler|	Helps CMake manage<br> mixed host/CUDA flags|
|-UCUTLASS_ENABLE_SYNCLOG|Undefine the macro|Controls conditional compilation<br> of CUTLASS synchronization logging|
<pre>
A distinction worth understanding is that the host compiler and GPU assembler have their own
optimization processes. -O3 does not expose a single optimization setting for every CUDA
compilation stage.

Also, -DNDEBUG affects assert() behavior. It should not be confused with removing all CUDA 
runtime error checking.
    
3.2 The most important flag: compute_90a and sm_90a
The command includes:
--generate-code=arch=compute_90a,code=[compute_90a,sm_90a]
This tells nvcc to generate both:
</pre>
|Target|Output|Purpose|
|:-----|:-----|:------|
|compute_90a|Architecture-specific PTX|Intermediate virtual GPU<br> instruction representation|
|sm_90a|Compiled GPU machine code|Native Hopper instructions<br> executable on H100|
<pre>
The important distinction is:
compute_90a describes the virtual instruction target; sm_90a describes the actual GPU code
generation target.
The compiler first lowers CUDA C++ to lower-level device instructions, then ptxas assembles 
and optimizes code for Hopper.

NVIDIA documents the distinction between virtual and real GPU architectures and the use of both
PTX and machine-code images in the executable. 

3.3 Why is 90a different from 90?
The a suffix identifies architecture-specific features that are not guaranteed to work on later
GPU architectures.
The WGMMA instruction family is one of the reasons for targeting sm_90a.
We should not think of it as just a newer spelling of sm_90.
H100 — Hopper  sm_90a compatible
RTX 5080 — Blackwell   Cannot execute SM90a WGMMA
This is precisely why my desktop RTX 5080 Blackwell SM120 can compile the executable but cannot
run that Hopper WGMMA kernel.
The compiled machine code is for the target architecture, not the GPU doing the compilation.
    
3.4 Why include both PTX and machine code?
The build retains the architecture-specific PTX in addition to generating the sm_90a machine
code.
This can be useful for inspecting the intermediate representation and for compatible driver JIT
scenarios.
However, the compute_90a PTX is not a general forward-compatible fallback for the Blackwell RTX
5080 SM120. Architecture-specific PTX features are restricted to compatible targets.
For the H100, the driver can use the native sm_90a machine code embedded in the executable.
    
3.5 --generate-line-info: Essential for the Nsight studies
The compiler command contains:
--generate-line-info
This generates device-code source-line information.
That is especially useful because we want to connect:
CUDA C++ source
       |
       V
PTX
       |
       V
SASS
       |
       V
Profiler metrics

The earlier Nsight Compute screenshots displayed source references such as wgmma_sm90.cu:235
and axpby.hpp:90.
That is consistent with having useful line-number information in the binary.
One important distinction: line information does not necessarily embed the full source text.
For source-level browsing on a different computer, the source files may still need to be 
accessible or embedded during profiling.
NVIDIA recommends line information rather than disabling device optimization with -G for
performance profiling. 

3.6 -Xptxas=-v and --resource-usage
These two options explain why your build prints detailed resource information.
-Xptxas=-v
--resource-usage
-Xptxas forwards the specified option to the PTX assembler.
-v requests verbose assembler output.
--resource-usage asks nvcc to show information such as register and memory usage. For a 
non-relocatable-device-code build, this also enables verbose assembler reporting, so there is
some redundancy. 
That is how we obtained:
Used 114 registers, used 1 barriers

Without verbose resource reporting, we might still compile successfully but not see these
valuable details.
3.7 -Xptxas=-warn-spills
This enables warnings about register spilling.
It does not prevent spilling.
Register spilling occurs when the compiler needs to place values in memory because they cannot
all remain in physical registers.
For GPU optimization, this matters because spilled values can introduce additional loads and
stores, often involving local memory.
We will examine the actual spill counts shortly.
    
3.8 --options-file, -MD, -MT, and -MF
These are primarily build-system options.
--options-file reads additional compilation options from the response file generated by CMake,
often including a long list of header search paths.
-MD, -MT, and -MF are used to generate and manage dependency information so the build system
knows when changing a header requires recompilation.
For example, if we edit an included CUTLASS header, CMake and Make need to know that
wgmma_sm90.cu.o may be out of date.
These options do not themselves change the CUDA kernel's tile shape or pipeline behavior.
    
3.9 -x cu, -c, and -o
-x cu -c wgmma_sm90.cu -o wgmma_sm90.cu.o

These specify:
- -x cu: Treat the input as CUDA source.
- -c: Compile without producing the final linked executable.
- -o: Write the resulting object file to the given location.
This completes the object-compilation stage.

4. Understand ptxas info: 47 bytes gmem
The output contains:
ptxas info : 47 bytes gmem
This is not the amount of global memory the GEMM uses to store A, B and C.
It describes statically allocated global-memory storage associated with the compiled device
module.
NVIDIA explicitly distinguishes module-level global-memory allocations from per-kernel resource
usage in its compiler documentation. 
The actual GEMM matrices are dynamically allocated at runtime.
For a 5120 x 5120 x 4096 FP16 GEMM, the matrix data alone is vastly larger than 47 bytes.
Therefore, this line tells us little about your GEMM's memory bandwidth or memory footprint.

5. Why does the compiler build cub::EmptyKernel?
The log shows:
Compiling entry function
'_ZN3cub17CUB_200802_SM_90011EmptyKernelIvEEvv'
for 'sm_90a'
This is a CUB library-generated CUDA kernel.
CUB is part of NVIDIA's CUDA ecosystem and provides reusable GPU primitives.
The application includes CUDA-related template libraries, and some of their generated code 
can produce additional kernel entry points.

Its resource usage is:
0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
Used 4 registers, used 0 barriers

This is not the WGMMA matrix multiplication kernel.
We should not compare its four registers with the 114 registers reported for the GEMM.
It is simply another device entry function that the compiler generated.

6. Why does ptxas compile gemm_device twice?
The assembler reports two device kernels:
</pre>
|Specialization|Registers/thread|Barriers|Assembler time|
|:-------------|:---------------|:-------|:-------------|
|TN GEMM — K-major layout|112|1|57.885 ms|
|NT GEMM — MN-major layout|114|1|51.020 ms|
<pre>
Both target sm_90a.
    
6.1 The source uses C++ templates
The kernel is declared approximately as:
template <
    class ProblemShape,
    class CtaTiler,
    class TA,
    class AStride,
    class ASmemLayout,
    class TiledCopyA,
    class TB,
    class BStride,
    class BSmemLayout,
    class TiledCopyB,
    class TC,
    class CStride,
    class TiledMma,
    class Alpha,
    class Beta
>
__global__ void gemm_device(...);
This is a template, not a single fixed kernel implementation.
When different template arguments are supplied, the compiler can generate different machine-
code specializations.
The source contains two host-side configurations:

NT GEMM:
auto dA = make_stride(Int<1>{}, ldA);
auto dB = make_stride(Int<1>{}, ldB);

auto sA = tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<TA>{},
    make_shape(bM, bK, bP));

TiledMMA tiled_mma = make_tiled_mma(
    SM90_64x64x16_F16F16F16_SS<
        GMMA::Major::MN,
        GMMA::Major::MN>{});


TN GEMM:
auto dA = make_stride(ldA, Int<1>{});
auto dB = make_stride(ldB, Int<1>{});

auto sA = tile_to_shape(
    GMMA::Layout_K_SW128_Atom<TA>{},
    make_shape(bM, bK, bP));

TiledMMA tiled_mma = make_tiled_mma(
    SM90_64x64x16_F16F16F16_SS<
        GMMA::Major::K,
        GMMA::Major::K>{});


These are distinct compile-time configurations.
Although both compute GEMM, they use different memory layouts and WGMMA operand major modes.
That can change address-generation instructions, copy partitioning, register lifetimes, and 
other generated code.
That explains why one specialization requires 112 registers while the other requires 114.

6.2 How can we identify the specializations in the mangled names?
The first specialization contains a stride type corresponding to:
make_stride(ldA, Int<1>{})
That is the TN path, where K is the contiguous dimension.
    
The second specialization contains a stride type corresponding to:
make_stride(Int<1>{}, ldA)
That is the NT path, where M is the contiguous dimension.

The long mangled symbols also encode CuTe layouts, swizzle types, and MMA atoms.
For example:
SwizzleILi3ELi4ELi3E
corresponds to the compile-time swizzle type:
Swizzle<3,4,3>
And:
MMA_64x64x16_F16F16F16_SS
represents the selected Hopper WGMMA atom.
The enormous kernel names result from C++ template instantiation and name mangling. 
They are not errors.
    
6.3 Why compile both if we normally execute NT?
The host program supports both configurations through runtime arguments:
if (transA == 'N' && transB == 'T') {
    return gemm_nt(...);
}
else if (transA == 'T' && transB == 'N') {
    return gemm_tn(...);
}
Both call paths exist in the compiled application.
Consequently, both template specializations are instantiated and compiled, even if we normally 
run with the default N T configuration.
At runtime, the application chooses the appropriate specialization.

This distinction is important:
- Tile shape and CuTe layouts are compile-time specializations.
- GEMM dimensions m, n, and k are runtime values.
- The NT versus TN execution path is selected using the runtime arguments.
Changing the matrix dimensions does not necessarily require recompiling the program.

7. Understand the resource usage of the actual GEMM kernel
Now focus on these two outputs:
Function properties for gemm_device ... TN
0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
Used 112 registers, used 1 barriers

Function properties for gemm_device ... NT
0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
Used 114 registers, used 1 barriers


7.1 Used 114 registers — What does that mean?
A CUDA thread has access to a set of physical registers.
The compiler uses them to hold temporary values, including:
- Address calculations.
- Loop counters.
- Tensor coordinate information.
- Accumulator values.
- Intermediate computation results.
- Synchronization and pipeline bookkeeping.
The default NT specialization requires:
114 registers per thread

With 128 threads per CTA:
114 x 128 =14,592 registers
That is the unrounded logical register demand per CTA.
The hardware allocates registers at a defined granularity, so the actual allocation can be 
slightly larger.
    
7.2 Connect this to the Nsight Compute report
The previous Launch Statistics screenshot reported:
Registers Per Thread: 114
The assembler reports:
Used 114 registers
Local compiler — ptxas114
registers/thread
H100 — Nsight Compute114
registers/thread

The profiler's register count matches the NT kernel compiled on my desktop RTX 5080 
Blackwell SM120.
This is a particularly useful verification.
It supports the conclusion that the H100 executed the expected specialization with the 
resource usage reported by my local compiler.
    
7.3 Is 114 registers per thread too high?
Not necessarily.
An advanced Tensor Core GEMM can deliberately use many registers to keep its accumulators and
intermediate values close to the compute units.
High register usage is not inherently bad.
It becomes a problem when it restricts occupancy enough to prevent the GPU from hiding 
important latencies.

But The Nsight Compute Occupancy section reported:
Resource	Maximum resident CTAs
Registers	4
Shared memory	2
This is important.
Even though the kernel uses 114 registers per thread, register usage is not the resource
limiting the current kernel to two CTAs per SM.
Shared memory is.
Therefore, reducing register usage alone is unlikely to improve your current theoretical
occupancy unless another resource requirement also changes.

8. Your zero register spills are a positive result
Your compiler reports:
0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
Let's interpret these carefully.
    
8.1 Stack frame: 0 bytes
The compiler reports no per-thread stack-frame allocation for these kernel specializations.
This means it does not require a stack frame for the compiled kernel in the way reported by ptxas.
It does not prove that every possible memory access uses registers or shared memory.
**************************************************
A stack frame for the compiled kernel:
In your compilation output:
ptxas info:
    0 bytes stack frame
    0 bytes spill stores
    0 bytes spill loads

0 bytes stack frame means the compiler did not need to allocate a per-thread stack frame for that GPU kernel.
To understand this, we need to distinguish registers, local memory, and the stack in CUDA.
1. What is a stack frame?
In C++, when a function is called, the compiler may need somewhere to store information associated with that 
particular function invocation.
Examples include:
- Local variables that cannot be kept in registers.
- Certain temporary objects and arrays.
- Saved information needed when calling another function.
- Information needed to return to the calling function.
    
The memory region reserved for one function invocation is called a stack frame.
For example, imagine a CPU function:
void calculate(int x) {
    int a = x + 10;
    int b = a * 2;
}

Conceptually, a compiler could use a stack frame to hold a and b.
CPU memory stack

       Higher addresses
      ┌─────────────────┐
      │ Calling function│
      ├─────────────────┤
      │ Return data     │
      ├─────────────────┤
      │ Local variable a│
      ├─────────────────┤
      │ Local variable b│
      └─────────────────┘
       Lower addresses

However, an optimizing compiler would likely keep a and b in registers instead, and might eliminate some 
calculations entirely.
A function does not automatically require a nonzero stack frame just because it contains local variables.
    
2. How does this work on a CUDA GPU?
CUDA is similar conceptually, but GPU stack storage is different from CPU stack storage.
Each CUDA thread has its own execution state and can have its own stack.
When a GPU thread needs stack storage, that storage is part of its per-thread local-memory address space, rather than 
shared memory.
Simplified CUDA thread storage
Registers:
Fast storage for values such as indices, pointers, and accumulators
    
Local memory / thread stack:
Per-thread address space that can hold stack frames, certain local objects, and spilled values

Shared memory:
Storage shared among threads in a CTA, such as your staged matrix tiles sA and sB

CUDA local memory is logically private to a thread, but physically resides in device memory and is accessed through 
the GPU cache hierarchy.
A GPU thread does not need to place every C++ local variable on the stack. The compiler tries to use registers 
when practical.
3. Example: When a CUDA function might need a stack frame
Consider:
__device__ __noinline__
float process(int index) {
    float values[128];
    for (int i = 0; i < 128; ++i) {
        values[i] = i * 0.5f;
    }
    return values[index];
}
__global__ void kernel(float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = process(tid % 128);
}

Notice two things:
1. process() is a separate device function, with inlining explicitly disabled.
2. values[index] uses a runtime index.
The compiler may need addressable per-thread storage for values, rather than keeping everything in registers.
A possible implementation requires stack storage for the function's local data.
Conceptually, for one CUDA thread:
Thread 0
┌─────────────────────────┐
│ kernel() invocation     │
│                         │
│ Calls process()         │
├─────────────────────────┤
│ process() stack frame   │
│                         │
│ Local array storage     │
│ Function call state     │
└─────────────────────────┘

Thread 1
┌─────────────────────────┐
│ kernel() invocation     │
│                         │
│ Calls process()         │
├─────────────────────────┤
│ process() stack frame   │
│                         │
│ Local array storage     │
│ Function call state     │
└─────────────────────────┘

Each thread's stack storage is logically independent.
The exact stack allocation depends on compiler optimization and generated code. This example illustrates why stack storage
might be needed; it does not guarantee a particular ptxas stack-frame size.
        
4. Now return to the wgmma_sm90.cu
the compiler reports:
Function properties for gemm_device:
    0 bytes stack frame
    0 bytes spill stores
    0 bytes spill loads
    Used 114 registers, used 1 barriers

Let's interpret each line separately.
Compiler output	Meaning
0 bytes stack frame	No per-thread stack frame required by this compiled kernel
0 bytes spill stores	No compiler-generated register-spill stores
0 bytes spill loads	No compiler-generated register-spill loads
Used 114 registers	Each thread requires 114 registers
Used 1 barriers	One hardware barrier resource reported for the kernel

The key point is that these describe different resources.
The WGMMA kernel has many C++ variables and CuTe tensor objects, but many are compile-time abstractions or are optimized 
into registers and address calculations.

For example:
Tensor tCrC = thr_mma.partition_C(gC);
This defines a CuTe tensor view. It does not automatically mean the compiler creates a stack allocation for the entire tensor.
Likewise, the accumulator fragment may occupy registers without needing stack storage.
And the shared-memory tensors sA and sB reside in shared memory. They are not per-thread stack frames.
        
5. Does a zero-byte stack frame mean the kernel never accesses local memory?
Not necessarily.
This is an important distinction:
- Stack-frame allocation is one reason a kernel may use local memory.
- Register spilling is another.
- Certain thread-local arrays or compiler-generated objects may also use local memory.
Therefore, 0 bytes stack frame by itself does not prove that all local-memory traffic is absent.
However, combining a zero-byte stack frame with zero spill loads and stores is encouraging: the compiler is not 
reporting those common sources of additional per-thread memory traffic.
        
6. Why does this matter for GPU performance?
If a kernel needs extra local-memory accesses, those accesses can introduce memory traffic and latency.
By contrast, registers provide very fast operand access.
The WGMMA kernel uses 114 registers per thread without reported spills. That is generally preferable to a version 
that spills frequently.

High register usage can reduce occupancy. In the particular Nsight Compute report, shared memory—not registers—was 
the resource limiting occupancy to 12.5%.
The distinction we should know and remember:
A CUDA thread can have local variables without having a stack frame.
A C++ local variable is defined by its scope. That does not determine its physical storage location.
The compiler decides whether a value is held in a register, placed in memory, or optimized away.
And a stack frame is just one way the compiler can organize per-thread memory needed for a function invocation.
That is exactly what 0 bytes stack frame tells us: for the compiled gemm_device kernel, ptxas did not require such a frame.
    
**************************************************
8.2 Spill stores: 0 bytes
The compiler did not generate register spill stores for these specializations.
8.3 Spill loads: 0 bytes
The compiler did not generate register spill loads.
Taken together:
The GEMM successfully keeps its compiler-managed register values in registers without register spilling.
That is useful because spills could add substantial local-memory traffic.
        
8.4 Connect this to the Long Scoreboard stalls
Earlier, the Nsight Compute report showed approximately:
Long Scoreboard: 8.3 cycles per issued instruction
One possible explanation for memory-dependency stalls in some kernels is register spilling.
But the compiler reports zero spills.
Therefore, we can rule out compiler-generated register spilling as the direct explanation for the kernel's 
high Long Scoreboard stall count.
We should instead investigate the kernel's actual memory instructions and their consumers.
This does not prove that every long-scoreboard stall is caused by cp.async; the dependency attribution still needs 
source-level analysis.

9. What does used 1 barriers mean?
The compiler reports:
Used 1 barriers
This is a per-kernel resource-usage statistic.
It means the compiled kernel reserves one hardware barrier resource of the type counted by ptxas.
It does not mean:
- The kernel performs only one synchronization operation.
- The kernel has only one pipeline stage.
- The kernel executes __syncthreads() exactly once.
- WGMMA executes only once.
A kernel can reuse a barrier many times during execution.
For example:
for (int k = 0; k < K; ++k) {
    // Work performed by several warps
    __syncthreads();
}
The barrier instruction can execute repeatedly even if only one barrier resource is allocated.
Likewise, Hopper's WGMMA synchronization mechanisms should not be treated as identical to the resource count displayed 
in this particular assembler statistic.
The source uses asynchronous copies and WGMMA coordination. The compiler's resource number does not describe the total
dynamic synchronization activity.

10. Why is shared-memory usage missing from the compiler output?
The assembler output reports registers and barriers, but does not print the 96 KiB of shared memory we saw in Nsight Compute.
That is expected because the source allocates its A/B pipeline storage dynamically.
The kernel uses a declaration like:
extern __shared__ char shared_memory[];
Then the host-side launch supplies a shared-memory size:
int smemBytes =
    sizeof(SharedStorage<
        TA, TB,
        decltype(sA),
        decltype(sB)>);
The compiler knows the shared-memory layout types, but the allocation is requested as dynamic shared memory 
at launch time.
The assembler's static shared-memory resource report is therefore not a complete representation of the actual 
shared-memory allocation used by the launch.
The Nsight Compute screenshot gives that value:
Dynamic Shared Memory Per Block: 98.30 KB
We already derived it:
3 x (128 x 64+128 x 64) x 2
=98,304 bytes
This is one of the reasons you should use both compiler resource reports and runtime profiler information.

11. The most important optimization insight from the compiler output
Let's combine the compilation results with the H100 measurements.
Property                      Source of information    Value
Registers/thread              ptxas                    114
Register spills               ptxas                    0
Threads/CTA                   Nsight Compute           28
Dynamic shared memory/CTA    Nsight Compute            96 KiB
Theoretical occupancy        Nsight Compute            12.50%
Achieved occupancy           Nsight Compute            12.42%
Eligible warps/scheduler     Nsight Compute            0.15
Long-scoreboard stalls       Nsight Compute           ~8.3 cycles/instruction

Here is the combined reasoning:
First, the compiler produces an optimized Hopper kernel using 114 registers per thread with no register spills.
Second, the launch uses three shared-memory stages, consuming approximately 96 KiB per CTA.
Third, shared-memory resource limits restrict the H100 to two resident CTAs per SM.
Fourth, those two CTAs provide eight resident warps, corresponding to 12.5% theoretical occupancy.
Fifth, the schedulers frequently cannot issue new instructions, and memory dependencies account for a large 
portion of the observed stalls.
This creates an interesting research question:
Would reducing the shared-memory pipeline from three stages to two increase resident CTAs sufficiently to 
improve latency hiding, or would losing a stage reduce copy/computation overlap enough to make performance worse?
We can estimate the resource tradeoff before changing the code.
Assuming the same tile dimensions and roughly the same other resources:
Stages    A/B shared memory   Potential resident CTAs/SM     Theoretical occupancy
2         64 KiB              Up to 3                        Up to 18.75%
3         96 KiB              2                              12.50%
4         128 KiB             1                              6.25%

These are resource-model estimates, not measured performance results.

The actual compiler register count and synchronization implementation can change when we modify the number of stages.
Also, higher occupancy does not guarantee higher GEMM throughput. More stages may enable better prefetching and
memory/computation overlap, while fewer stages may increase residency.
This is a good candidate for a controlled experiment later.

12. Understand the compilation-time numbers
The output contains:
Compile time = 7.958 ms    // CUB EmptyKernel
Compile time = 57.885 ms   // TN GEMM
Compile time = 51.020 ms   // NT GEMM
        
These are ptxas compilation times for the individual functions.
They are not the total time required to build the entire CUDA program.

The whole build includes:
- CMake dependency checks.
- CUDA C++ parsing.
- Template instantiation.
- Host compilation.
- Device-code generation.
- PTX assembly.
- Host linking.
CUTLASS and CuTe use substantial template metaprogramming, so the total build time can be much greater than 
the short assembler times shown here.
For reference, 57.885 ms is approximately 0.058 seconds.
It is a compilation measurement, not a kernel execution measurement.

13. Understand the final linking step
The log finishes with:
[100%] Linking CUDA executable wgmma_sm90
Then:
/usr/bin/g++ \
    @CMakeFiles/wgmma_sm90.dir/objects1.rsp \
    -o wgmma_sm90 \
    @CMakeFiles/wgmma_sm90.dir/linkLibs.rsp \
    -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" \
    -L"/usr/local/cuda/targets/x86_64-linux/lib"

The linker combines the object files and required libraries.
The response files supply lists of objects and link options without putting everything directly on one enormous 
command line.
The CUDA library paths tell the linker where to find the required CUDA-related libraries.
The final result is:
[100%] Built target wgmma_sm90

The executable is located at:
/home/xxx/cuda-learning/cute_hopper/lambda_sm90/build/wgmma_sm90
This executable includes the host-side application along with device-code images for the requested GPU architecture.
That is the file has been successfully transferred to Lambda and executed on the H100.

14. Inspect the actual Hopper PTX and SASS on the desktop
This can  be done one step deeper without renting another H100.
The CUDA Toolkit includes cuobjdump, which can inspect device code embedded in a compiled executable.

On the desktop RTX 5080 Blackwell SM120, execute:
Inspect PTX
cuobjdump --dump-ptx build/wgmma_sm90 > wgmma_sm90.ptx.txt
Search for WGMMA and asynchronous copies:
grep -nE 'wgmma|cp\.async' wgmma_sm90.ptx.txt | head -n 40
This helps us identify the virtual assembly instructions that the CuTe code generated.
Inspect SASS
cuobjdump --dump-sass build/wgmma_sm90 > wgmma_sm90.sass.txt

Then:
grep -nE 'HGMMA|LDGSTS|WARPGROUP|BAR' \
    wgmma_sm90.sass.txt | head -n 40

These instructions correspond to important parts of the source code:
CUDA/CuTe operation            Lower-level instruction family
cp.async copy                  LDGSTS
cute::gemm(mma, ...)           HGMMA
Warpgroup synchronization      WARPGROUP-related SASS
CTA synchronization            Barrier-related SASS
        
Exact instruction spellings and counts depend on the compiler version and generated code.
This is particularly valuable: We can inspect both the Hopper PTX and its compiled SASS entirely on the 
RTX 5080 desktop. We only need an H100 to execute and profile the code.
        
15. Final conclusions from the build
Finding                                Significance
sm_90a machine code generated          Correct native target for your H100 WGMMA experiment
PTX also retained                      We can inspect the intermediate code
Two GEMM specializations generated     NT and TN have different compiled implementations
114 registers for NT                   Matches the actual H100 profiler result
No register spills                     No compiler-generated spill traffic in these kernels
One barrier resource                   Synchronization-related resource allocation
96 KiB dynamic shared memory           Explains the shared-memory occupancy restriction
Line information enabled               Supports source-to-SASS investigation in Nsight Compute
Linking successful                     Executable is ready for SCP and execution


The most valuable next step is to connect three pieces of evidence:
CUDA source line → generated SASS instruction → Nsight Compute stall metric.
For example, find the LDGSTS instructions generated by the CuTe asynchronous copies and examine which dependent 
instructions are associated with your Long Scoreboard stalls.
That investigation can help determine whether the current three-stage design provides enough 
copy/computation overlap.

The key takeaway from this compilation output: The kernel is not obviously wasting performance through register 
spilling. Its measured performance limitation is more closely connected to the combination of shared-memory 
residency, synchronization, and memory-dependency latency. The compiler report tells us what resources the 
kernel requires; Nsight Compute tells us how effectively the H100 uses those resources.

This completes the compilation-output analysis and connects it to the Launch Statistics and Occupancy 
screenshots.





