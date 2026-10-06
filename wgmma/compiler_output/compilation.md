<pre>
First build output gemm_nt:
cmake --build build
[ 50%] Building CUDA object CMakeFiles/wgmma_sm90.dir/wgmma_sm90.cu.o
ptxas info    : (C7510) Potential Performance Loss: wgmma.mma_async instructions are serialized due to wgmma pipeline crossing function boundary at a function call in the function '_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJiNS3_ILi1EEEEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJNS3_ILi8EEENS3_ILi16EEEEEENS1_IJS5_S9_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS5_NS3_ILi512EEEEEENS1_IJS9_NS3_ILi0EEEEEENS1_IJSQ_NS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES10_EES8_EEENSG_INS1_IJSJ_SH_EEENS1_IJNS1_IJS4_S9_EEESI_EEEEENS1_IJSI_S5_EEEEES8_SA_SW_S18_S8_NS1_IJS9_iEEENS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1D_5MajorE0ELS1F_0ELNS1D_7ScaleInE1ELS1G_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSQ_SQ_SQ_EEEEENS1_IJNS0_10UnderscoreES1M_S1M_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_'
ptxas info    : (C7510) Potential Performance Loss: wgmma.mma_async instructions are serialized due to wgmma pipeline crossing function boundary at a function call in the function '_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJNS3_ILi1EEEiEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJS5_NS3_ILi2EEEEEENS1_IJNS3_ILi8EEESJ_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS9_NS3_ILi512EEEEEENS1_IJS5_NS3_ILi1024EEEEEENS1_IJNS3_ILi0EEENS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES11_EES8_EEENSG_INS1_IJS4_SJ_EEENS1_IJSJ_S9_EEEEES14_EES8_SA_SX_S17_S8_SA_NS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1B_5MajorE1ELS1D_1ELNS1B_7ScaleInE1ELS1E_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSS_SS_SS_EEEEENS1_IJNS0_10UnderscoreES1K_S1K_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_'
ptxas info    : 1076 bytes gmem
ptxas info    : Compiling entry function '_ZN3cub17CUB_200802_SM_90011EmptyKernelIvEEvv' for 'sm_90a'
ptxas info    : Function properties for _ZN3cub17CUB_200802_SM_90011EmptyKernelIvEEvv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 4 registers, used 0 barriers
ptxas info    : Compile time = 0.870 ms
ptxas info    : Compiling entry function '_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJiNS3_ILi1EEEEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJNS3_ILi8EEENS3_ILi16EEEEEENS1_IJS5_S9_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS5_NS3_ILi512EEEEEENS1_IJS9_NS3_ILi0EEEEEENS1_IJSQ_NS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES10_EES8_EEENSG_INS1_IJSJ_SH_EEENS1_IJNS1_IJS4_S9_EEESI_EEEEENS1_IJSI_S5_EEEEES8_SA_SW_S18_S8_NS1_IJS9_iEEENS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1D_5MajorE0ELS1F_0ELNS1D_7ScaleInE1ELS1G_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSQ_SQ_SQ_EEEEENS1_IJNS0_10UnderscoreES1M_S1M_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_' for 'sm_90a'
ptxas info    : Function properties for _Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJiNS3_ILi1EEEEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJNS3_ILi8EEENS3_ILi16EEEEEENS1_IJS5_S9_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS5_NS3_ILi512EEEEEENS1_IJS9_NS3_ILi0EEEEEENS1_IJSQ_NS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES10_EES8_EEENSG_INS1_IJSJ_SH_EEENS1_IJNS1_IJS4_S9_EEESI_EEEEENS1_IJSI_S5_EEEEES8_SA_SW_S18_S8_NS1_IJS9_iEEENS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1D_5MajorE0ELS1F_0ELNS1D_7ScaleInE1ELS1G_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSQ_SQ_SQ_EEEEENS1_IJNS0_10UnderscoreES1M_S1M_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 116 registers, used 1 barriers
ptxas info    : Compile time = 54.434 ms
ptxas info    : Compiling entry function '_Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJNS3_ILi1EEEiEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJS5_NS3_ILi2EEEEEENS1_IJNS3_ILi8EEESJ_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS9_NS3_ILi512EEEEEENS1_IJS5_NS3_ILi1024EEEEEENS1_IJNS3_ILi0EEENS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES11_EES8_EEENSG_INS1_IJS4_SJ_EEENS1_IJSJ_S9_EEEEES14_EES8_SA_SX_S17_S8_SA_NS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1B_5MajorE1ELS1D_1ELNS1B_7ScaleInE1ELS1E_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSS_SS_SS_EEEEENS1_IJNS0_10UnderscoreES1K_S1K_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_' for 'sm_90a'
ptxas info    : Function properties for _Z11gemm_deviceIN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJNS3_ILi1EEEiEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi4ELi3EEENS0_18smem_ptr_flag_bitsILi16EEENS0_6LayoutINS1_IJNS1_IJS5_NS3_ILi2EEEEEENS1_IJNS3_ILi8EEESJ_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJS9_NS3_ILi512EEEEEENS1_IJS5_NS3_ILi1024EEEEEENS1_IJNS3_ILi0EEENS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES11_EES8_EEENSG_INS1_IJS4_SJ_EEENS1_IJSJ_S9_EEEEES14_EES8_SA_SX_S17_S8_SA_NS0_8TiledMMAINS0_8MMA_AtomIJNS0_4SM904GMMA25MMA_64x64x16_F16F16F16_SSILNS1B_5MajorE1ELS1D_1ELNS1B_7ScaleInE1ELS1E_1EEEEEENSG_INS1_IJS9_S9_S9_EEENS1_IJSS_SS_SS_EEEEENS1_IJNS0_10UnderscoreES1K_S1K_EEEEES8_S8_EvT_T0_PKT1_T2_T3_T4_PKT5_T6_T7_T8_PT9_T10_T11_T12_T13_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 116 registers, used 1 barriers
ptxas info    : Compile time = 53.070 ms
[100%] Linking CUDA executable wgmma_sm90
[100%] Built target wgmma_sm90
</pre>


<pre>
A tiny dummy kernel from CUB:
cub::CUB_200802_SM_900::EmptyKernel<void>
The program never calls CUB itself. It does include it indirectly, though. Any translation unit (one .cu file plus everything it includes) that pulls in CUB's headers gets this kernel compiled into it.

Where CUB comes from

wgmma_sm90.cu stores its matrices in Thrust containers. In main it creates the host data in thrust::host_vector<TA> h_A and so on, then copies it to the GPU with thrust::device_vector<TA> d_A = h_A. Thrust's GPU backend is built on CUB, so the include chain is roughly:

wgmma_sm90.cu
 └─ <thrust/device_vector.h>
     └─ thrust CUDA backend headers
         └─ <cub/util_device.cuh>   ← defines EmptyKernel

CUTLASS's own utility headers can also pull CUB in. Either way, it comes in through headers, not through code you wrote.

We can confirm this it by asking nvcc for the list of every header the file includes:

nvcc -std=c++17 -arch=sm_90a \
  -I../include \
  -I../cutlass/include \
  -I../cutlass/tools/util/include \
  -M wgmma_sm90.cu | tr ' ' '\n' | grep -v -E '^\\?$' > deps.txt

-M makes nvcc print the dependency list without compiling. You should see cub/util_device.cuh in the output.
util_device.cub appear in deps.txt
/usr/local/cuda/bin/../targets/x86_64-linux/include/cub/util_device.cuh

What EmptyKernel is in the cub/util_device.cuh it is essentially:
template <typename T>
__global__ void EmptyKernel() {}
It does nothing. CUB uses it as a probe: at runtime, CUB calls cudaFuncGetAttributes(&attrs, EmptyKernel<void>) and reads attrs.ptxVersion, which tells it which architecture the binary was compiled for. That is how cub::PtxVersion() works, and CUB algorithms use it to choose tuning parameters. Because the header refers to EmptyKernel<void>, the template is instantiated in every translation unit that includes the header. ptxas then compiles it and reports it like any other kernel.

Decoding the namespace cub::CUB_200802_SM_900
CUB doesn't put its symbols directly in namespace cub. Its CUB_NAMESPACE_BEGIN macro adds an inline namespace that encodes two things:

| Part | Meaning |
| :--- | :--- |
|200802|The CUB version, encoded as major·100000 + minor·100 + patch, so 2.8.2. This is the CCCL version bundled with your CUDA 12.8 toolkit. It also tells you this build used the toolkit's built-in CCCL, not a separate ~/cccl checkout.|
|SM_900|The architectures this translation unit was compiled for. It comes from __CUDA_ARCH_LIST__, so here it's 900 for sm_90a. If you compiled for sm_120, it would read SM_1200.|
    
The purpose is to prevent ODR (One Definition Rule) violations. If two libraries in the same program were compiled with different CUB versions or for different architectures, their CUB symbols get different mangled names, so the linker never mixes one library's copy with the other's.
</pre>



=======================================================================================
