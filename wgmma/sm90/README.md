<pre>
This repository explores deep hardware-level profiling and optimization of the standard NVIDIA WGMMA tutorial kernels.
https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/hopper/wgmma_sm90.cu
  
./wgmma_sm90 5120 5120 4096 N T    # NT kernel (default)
./wgmma_sm90 5120 5120 4096 T N    # TN kernel
</pre>
