<pre>
Detailed Nsight Compute analysis — Hopper wgmma_sm90.cu
The first 20 screenshots

We can reconstruct how the GEMM uses the H100, identify where execution stalls, and form specific hypotheses about 
why the kernel does not achieve peak Tensor Core performance.

The preliminary conclusion is:
The kernel performs FP16 WGMMA correctly, but its performance appears limited partly by memory-access latency and 
insufficient warp-level latency hiding—not by saturated HBM bandwidth.

There is also evidence of inefficient global stores. We need to investigate those separately rather than assume they 
explain all the lost performance.

Three things throughout this analysis:
- Measured: What Nsight Compute actually reports.
- Derived: What we can calculate from those measurements.
- Hypothesis: A possible bottleneck that requires further evidence.
Step 1. Understand precisely what we profiled
The screenshots show the following launch.
Property	Value
GPU	NVIDIA H100 80GB HBM3
Kernel	gemm_device
Grid dimensions	(40, 40, 1)
Block dimensions	(128, 1, 1)
Kernel duration	651.87 μs
Elapsed cycles	951,535
SM frequency	1.45 GHz

Let's translate this into CUDA execution terminology.
1.1 Number of CTAs
Your grid contains:
\[
40\times40\times1=\boxed{1600\text{ CTAs}}
\]

Each CTA has 128 CUDA threads:
\[
128/32=\boxed{4\text{ warps per CTA}}
\]

Therefore, the launch comprises:
\[
1600\times4=\boxed{6400\text{ CTA-level warps}}
\]

Those are the warps launched over the entire kernel, not the number resident simultaneously on the GPU.

Because this tutorial uses one 128-thread warpgroup per CTA, each CTA has exactly one warpgroup.
1.2 Verify the GEMM dimensions
NVIDIA's wgmma_sm90.cu defaults are:
int m = 5120;
int n = 5120;
int k = 4096;


Its default CTA tile dimensions are:
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<64>{};


These agree with your measured (40,40,1) grid:
\[
\frac{5120}{128}=40
\]

Thus, each CTA calculates one \(128\times128\) output tile, accumulating contributions from 64 K-tiles. The source uses three shared-memory pipeline stages. 
GitHub+1

1.3 Calculate the total floating-point operations
For GEMM:
\[
C=A B
\]

Using the standard convention of two FLOPs per multiply-add:
\[
\text{FLOPs}=2MNK
\]

Substituting your dimensions:
\[
=2(5120)(5120)(4096)
\]

\[
=\boxed{214,748,364,800\text{ FLOPs}}
\]

That is approximately 214.75 GFLOPs for one GEMM execution.
Step 2. Calculate your measured performance
Your first screenshot reports:
\[
t=651.87\ \mu s
\]

Converting to seconds:
\[
t=0.00065187\ s
\]

Now calculate throughput:
\[
P=\frac{\text{FLOPs}}{t}
\]

\[
P=\frac{214,748,364,800}{0.00065187}
\]

\[
\boxed{P\approx329.43\text{ TFLOP/s}}
\]

Nsight Compute throughput329.43
TFLOP/s

Profiled execution time651.87
microseconds



Earlier, your standalone program reported:
CUTE_GEMM: [378214.7] GFlop/s (0.5678) ms


That represents approximately 378.21 TFLOP/s.
Measurement	Time	Throughput
Standalone benchmark	567.8 μs	378.21 TFLOP/s
Nsight Compute capture	651.87 μs	329.43 TFLOP/s

The profiled launch took approximately 14.8% longer than the earlier standalone average. This does not necessarily mean the profiling tool slowed the kernel itself by that amount. The captures can differ in cache conditions, clock frequency, launch selection, and profiling/replay behavior.

For performance optimization, use your unprofiled benchmark to compare actual runtime, and Nsight Compute to understand hardware behavior. NVIDIA documents that collecting counters may require multiple kernel replay passes.

 
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/98974c2d-5772-4ded-965e-0236c73e5a56" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/8c770961-e5d1-48e3-acbe-b89fca78a7dd" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/f0f68aa2-a4cd-49f2-aad9-131f2750cc68" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/f3a321dc-7f3d-4388-82ae-5eba4816c98c" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/4f7ef757-21d5-40ab-82ae-1f90e9a0c949" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/001ab819-58ad-4e76-b451-a5cbf1d03ef8" />


<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/2b9ede8b-2aba-4769-9163-348fd6f80cf0" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/948b4c7e-133f-4df5-9806-58c76b9f479b" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/6adac94a-0bbb-41c8-b51d-17dca6c43330" />

 <img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/f28b181e-afd2-4380-a4d7-58e8653f1b5e" />
 

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/fc442123-9e32-414d-ad24-78221f809cf7" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/07bbe929-ac2a-4b86-8e00-443fa1be0ea5" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/698167a0-609a-4620-8b1f-28f53b61ab14" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/95dc1779-539f-40e8-bc2e-1571e4a9c881" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/2f5103d4-5284-4af4-8c61-c713309a4496" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/f28dfd2e-e9eb-4be7-94e1-367fe9322e68" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/496ab5d4-a958-4458-ae1b-52a0f5209553" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/313a47bb-d8a9-489b-aa32-2edc157d5b1d" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/2a6a9258-6fb0-41c9-b31c-c630f075f834" />

<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/d91c052a-1f3d-435c-a126-4bd4cb6bd21c" />

  
</pre>
