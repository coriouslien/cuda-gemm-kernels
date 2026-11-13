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
Property	             Value
GPU                   NVIDIA H100 80GB HBM3
Kernel                gemm_device
Grid dimensions       (40, 40, 1)
Block dimensions      (128, 1, 1)
Kernel duration       651.87 μs
Elapsed cycles        951,535
SM frequency          1.45 GHz

Let's translate this into CUDA execution terminology.
1.1 Number of CTAs
The  grid contains:
40 x 40 x 1= 1600  CTAs

Each CTA has 128 CUDA threads:
128/32= 4  warps per CTA

Therefore, the launch comprises:
1600 x 4= 6400 CTA-level warps
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

These agree with the measured (40,40,1) grid:
5120/128=40

Thus, each CTA calculates one 128 x 128  output tile, accumulating contributions from 64 K-tiles. 
The source uses three shared-memory pipeline stages. 

1.3 Calculate the total floating-point operations
For GEMM:
C=A x B

Using the standard convention of two FLOPs per multiply-add:
FLOPs=2MNK

Substituting your dimensions:
=2(5120)(5120)(4096)
=214,748,364,800 FLOPs

That is approximately 214.75 GFLOPs for one GEMM execution.
Step 2. Calculate your measured performance
Your first screenshot reports:
t=651.87 mu s

Converting to seconds:
t=0.00065187 s

Now calculate throughput:
P=FLOPs/t
P=214,748,364,800/0.00065187
P approx 329.43  TFLOP/s

Nsight Compute throughput 329.43 TFLOP/s
Profiled execution time 651.87 microseconds
The standalone program reported:
CUTE_GEMM: [378214.7] GFlop/s (0.5678) ms
That represents approximately 378.21 TFLOP/s.
Measurement	               Time	             Throughput
Standalone benchmark	      567.8 μs	         378.21 TFLOP/s
Nsight Compute capture	    651.87 μs	        329.43 TFLOP/s
 
The profiled launch took approximately 14.8% longer than the earlier standalone average. This does not necessarily 
mean the profiling tool slowed the kernel itself by that amount. The captures can differ in cache conditions, 
clock frequency, launch selection, and profiling/replay behavior.

For performance optimization, use the unprofiled benchmark to compare actual runtime, and Nsight Compute to 
understand hardware behavior. NVIDIA documents that collecting counters may require multiple kernel 
replay passes.

1. GPU Speed of Light Throughput 
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/98974c2d-5772-4ded-965e-0236c73e5a56" />
2. GPU Speed of Light Throughput
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/8c770961-e5d1-48e3-acbe-b89fca78a7dd" />
3. GPU Speed of Light Throughput
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/f0f68aa2-a4cd-49f2-aad9-131f2750cc68" />
4. GPU Speed of Light Throughput
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/f3a321dc-7f3d-4388-82ae-5eba4816c98c" />
5. GPU Speed of Light Throughput
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/4f7ef757-21d5-40ab-82ae-1f90e9a0c949" />
6. GPU Speed of Light Throughput
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/001ab819-58ad-4e76-b451-a5cbf1d03ef8" />

7. GPU Speed of Light Throughput
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/2b9ede8b-2aba-4769-9163-348fd6f80cf0" />

Metric	                             Value
Compute (SM) Throughput             42.29%
Memory Throughput                   72.77%
L1/TEX Cache Throughput             82.82%
L2 Cache Throughput                 48.23%
DRAM Throughput                     16.41%

These numbers are extremely important.
3.1 What does Compute (SM) Throughput = 42.29% mean?
1). What does 42.29% mean?
Nsight Compute reports:
Compute (SM) Throughput [%] = 42.29
Nsight Compute compares achieved utilization against the relevant peak hardware throughput.
A value of 42.29% means the reported compute-throughput metric reaches approximately 42% of its reference maximum.
It does not mean that only 42.29% of the SMs are running.
This measures how close the busiest relevant SM compute resource is to its peak sustained throughput, 
averaged over the kernel's execution.
It does not mean only 42.29% of the GPU's SMs are active. NVIDIA defines this as an aggregate throughput metric derived
from underlying hardware counters. 
The FP16 WGMMA operation counter also reports 42.29% of its dense HGMMA peak, which confirms that Tensor Core 
throughput is the relevant performance measure for this kernel.
 
2). Is 42.29% good?
As a rough learning-oriented guide—not an official NVIDIA grading scale:
SM throughput       General interpretation
Below 20%           Low; investigate underutilization
20–40%              Moderate
40–60%              Reasonable, with optimization potential
60–80%              Strong utilization
Above 80%           Very high utilization

These ranges aren't universal. A memory-bound kernel can be extremely well optimized while having low SM throughput.
For the compute-intensive WGMMA GEMM, however, higher Tensor Core throughput is desirable.
 
3). Why is your throughput not higher?
The earlier Nsight Compute measurements provide useful evidence.
Metric                      the result	    What it tells us
Compute (SM) Throughput	    42.29%	        Considerable compute headroom
Theoretical Occupancy	      12.50%	        Few resident warps
Eligible Warps/Scheduler	   0.15	          Very few ready instructions
No Eligible Warp	          85.12%	         Schedulers frequently cannot issue
L1/TEX Throughput	         82.82%	         High internal memory activity
DRAM Throughput	           16.41%	         HBM bandwidth is not saturated
 
The important relationship is:
The two questions are different:
- How many SMs are doing work?
- How close are the compute pipelines to their maximum throughput?
The reported value concerns throughput, not the percentage of SMs assigned work.

42.29% Compute (SM) Throughput is a reasonable result for your first Hopper WGMMA experiment, but it is not 
particularly high for a highly optimized Tensor Core GEMM. There is substantial room for improvement.
The important relationship is:
96 KiB shared memory per CTA
           |
           V
Only 2 resident CTAs per SM
           |
           V
12.5% theoretical occupancy
           |
           V
Limited ability to hide memory latency
 
Tensor Cores may not receive work continuously
This is a plausible explanation supported by the occupancy and stall metrics, although the exact limiting instruction 
still needs source-level investigation.
4). What throughput should you aim for?
The captured kernel achieved:
Current performance 329.4           Dense HGMMA peak utilization 42.29%
 TFLOP/s                            of measured reference peak

Instead of immediately targeting 80% or 90%, trying to improve the kernel to 50–60% of the dense HGMMA peak 
while maintaining correctness.
That is an experimental goal, not a prediction that the tutorial can necessarily reach it.
We could begin by comparing two versus three pipeline stages, examining long-scoreboard stalls, and investigating the 
output-store access pattern. Measure execution time as well as throughput, because optimizing one metric does not 
guarantee a faster kernel.
 
5). One important lesson
High occupancy and high SM throughput are not the same thing.
A kernel can have 12.5% occupancy yet achieve substantial Tensor Core throughput because each WGMMA instruction performs 
a large amount of arithmetic asynchronously.

The task as a GPU performance engineer is not simply to maximize occupancy. It is to keep the hardware performing 
useful work with as little waiting as possible.
The conclusion: the 42.29% is a credible starting point, not a poor result. The interesting part is that the Nsight 
report already provides clues about why it is 42.29% and which experiments might improve it.

3.2 What does Memory Throughput = 72.77% mean?
Memory Throughput summarizes activity across important memory-system resources.
We should not interpret it as:
The H100 is using 72.77% of its maximum HBM bandwidth.
The actual DRAM throughput is only 16.41%.
This is the first important distinction:
The memory system is significantly utilized, but external HBM bandwidth is not saturated.

 
3.3 Compare the memory levels
3.3 Compare the memory levels
L1/TEX     82.82%
L2 Cache   48.23%
DRAM       16.41%
Reported throughput relative to each unit's respective peak. These are not percentages of one common bandwidth.
Notice the ordering:
82.82% > 48.23% > 16.41%
This suggests that internal memory activity is more significant than external memory bandwidth pressure.
It does not yet prove that L1 is the final bottleneck. For that, we need the Memory Workload Analysis and 
scheduler-stall information.
3.4 Interpret NVIDIA's High Memory Throughput warning
Nsight Compute states that memory is more heavily utilized than compute and recommends examining memory 
access patterns.
This is a diagnostic clue.
The investigation should be:
1. Is DRAM bandwidth saturated? No.
2. Is the L1/TEX subsystem heavily utilized? Yes.
3. Are there inefficient memory accesses? The later screenshots suggest yes, especially global stores.
4. Are warps waiting for memory operations? Yes, strongly.
The fourth observation is especially important because it connects memory behavior to lost compute performance.

Step 4. Understand the Tensor Core Roofline
Screenshots 2–4 — Tensor Core Operations Roofline and operation tables
We selected the correct chart: Roofline Tensor Core.
4.1 The mathematics behind the roofline
The fundamental roofline equation is:

P less or equal min(P_peak,B_peak x I)
Where:
Symbol              	Meaning
P	                   Achieved computational throughput
P_peak	              Theoretical peak compute throughput
B_peak	              Peak memory bandwidth
I	                   Arithmetic intensity, in operations per byte
 
The chart's vertical axis is computational performance, while the horizontal axis is arithmetic intensity.
The diagonal lines represent memory-bandwidth limitations. The horizontal lines represent compute-throughput ceilings.
The Tensor Core chart includes different memory-hierarchy levels, so multiple points or roofs can describe the 
same kernel using different measured traffic.

4.2 Why are there three colored points?
The screenshot shows three measured points at approximately the same vertical position but different arithmetic 
intensities.
The hierarchical roofline analyzes computation relative to different memory traffic levels, principally L1, L2, and DRAM.
The calculation is conceptually:
I_level= Tensor Core operations / Bytes transferred at that memory level
For example:
I_L1 = Operations / L1 traffic 
I_DRAM = Operations / DRAM traffic
 
The kernel's compute throughput is the same in these representations, but the denominator changes.
That is why the points share essentially the same height while appearing at different horizontal locations.
Important: These are different views of one kernel, not three separate kernels.

4.3 Why is the measured point below the roofline?
The measured Tensor Core rate is approximately:
329.43 TFLOP/s
The important question is whether the distance to a particular ceiling represents unused compute throughput, 
memory bandwidth limitation, or instruction and dependency latency.

The other counters show:
- Moderate Tensor Core pipeline utilization.
- Low issue-slot utilization.
- Very few eligible warps.
- Large long-scoreboard stalls.

Together, these suggest that Tensor Cores are not being supplied with work continuously enough to approach their 
dense-operation ceiling.
This is more informative than simply saying the roofline point is below the maximum.
 
4.4 Understand the Op:hgmma Src:fp16 Sparsity:off row
The third screenshot contains: (GPU Speed of Light Throughput)
Counter                                  Value
Number of operations                     214,748,364,800
Operations per cycle                     228,673.44
Operations per second, in gigaops/s      329,433.33
Peak utilization                         42.29%
Peak operations per cycle                540,672

The peak utilization mathematically: 
228,673.44 / 540,672 x 100 = 42.29%
This is excellent confirmation: The Nsight Compute throughput is based on Hopper FP16 WGMMA operations.
Notice the exact operation count:
214,748,364,800=2MNK
The profiler's Tensor Core operation count agrees with the expected GEMM workload.
That establishes consistency between the algorithm and what the hardware counter reports.
 
4.5 Why does the general Src:fp16 row show 21.15%?
Your fourth screenshot reports:
Metric                       Value
FP16 operations              214,748,364,800
Achieved operations/cycle    228,673.44
Peak operations/cycle        1,081,344
Peak percentage              21.15%

Compare its peak with the non-sparse HGMMA peak:
1,081,344=2 x 540,672
The general FP16 maximum used for this row is twice the dense HGMMA FP16 peak, reflecting the higher supported 
sparse Tensor Core operation rate.
Your kernel uses:




 
 
 
The a suffix identifies architecture-specific features that are not guaranteed to work on later GPU architectures.
The WGMMA instruction family is one of the reasons for targeting sm_90a.
We should not think of it as just a newer spelling of sm_90.
H100 — Hopper                  sm_90a compatible
RTX 5080 — Blackwell           Cannot execute SM90a WGMMA
This is precisely why the desktop RTX 5080 Blackwell SM120 can compile the executable but cannot run that 
 Hopper WGMMA kernel.
The compiled machine code is for the target architecture, not the GPU doing the compilation.
 
3.4 Why include both PTX and machine code?
The build retains the architecture-specific PTX in addition to generating the sm_90a machine code.
This can be useful for inspecting the intermediate representation and for compatible driver JIT scenarios.
However, the compute_90a PTX is not a general forward-compatible fallback for your Blackwell RTX 5080. 
Architecture-specific PTX features are restricted to compatible targets.

 3.5 --generate-line-info: Essential for your Nsight studies
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

The earlier Nsight Compute screenshots displayed source references such as wgmma_sm90.cu:235 and axpby.hpp:90.
That is consistent with having useful line-number information in the binary.
One important distinction: line information does not necessarily embed the full source text. For source-level 
browsing on a different computer, the source files may still need to be accessible or embedded during profiling.

NVIDIA recommends line information rather than disabling device optimization with -G for performance profiling. 
NVIDIA CUDA Compiler Driver 13.5 documentation
 
3.6 -Xptxas=-v and --resource-usage
These two options explain why your build prints detailed resource information.
-Xptxas=-v
--resource-usage
For the H100, the driver can use the native sm_90a machine code embedded in your executable.
 

 


 

 

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
